"""Authenticated MR-FlowDPO browser UI, served beside the original toolkit.

Model work runs in CLI subprocesses. Opening the browser imports no torch and
shares Side-Step's operation mutex, bearer-token and loopback-host validation.
"""
from __future__ import annotations

import asyncio
from collections import deque
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

_ROOT = Path(__file__).resolve().parent.parent
_STORE = _ROOT / ".jk_step"
_ACTIVE_STATUSES = {"running", "stopping", "orphaned"}


def _jsonable(value: Any) -> Any:
    """Keep diagnostic/result data useful without assuming model types."""
    return json.loads(json.dumps(value, default=str, ensure_ascii=False),
                      parse_constant=lambda value: value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")
    temporary.replace(path)


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class JobManager:
    """One model operation at a time, with bounded logs and resumable configs."""

    def __init__(self, task_manager=None, root: Path | None = None) -> None:
        self.task_manager = task_manager
        self.root = Path(root or _ROOT)
        self.store = self.root / ".jk_step" / "jobs"
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._events: dict[str, deque] = {}
        self._sequence: dict[str, int] = {}
        if self.store.is_dir():
            paths = sorted(self.store.glob("*/job.json"), key=lambda path: path.stat().st_mtime)
            for path in paths[-30:]:
                try:
                    job = json.loads(path.read_text("utf-8"))
                    if job["status"] in _ACTIVE_STATUSES:
                        job["status"] = "orphaned" if _pid_running(int(job.get("pid", 0))) else "interrupted"
                    self._jobs[job["id"]] = job
                except (OSError, ValueError, KeyError, TypeError):
                    continue

    def active(self) -> dict | None:
        with self._lock:
            for job in self._jobs.values():
                if job["status"] == "orphaned" and not _pid_running(int(job.get("pid", 0))):
                    job["status"] = "interrupted"
                if job["status"] in _ACTIVE_STATUSES:
                    return dict(job)
        return None

    def _persist(self, job: dict) -> None:
        _write_json(self.store / job["id"] / "job.json", job)

    def _emit(self, job_id: str, event: dict) -> None:
        with self._lock:
            if "type" not in event and "event" in event:
                event = {**event, "type": event["event"]}
            number = self._sequence.get(job_id, 0) + 1
            self._sequence[job_id] = number
            entry = {**_jsonable(event), "sequence": number, "time": time.time()}
            self._events.setdefault(job_id, deque(maxlen=1500)).append(entry)
            job = self._jobs[job_id]
            if entry.get("type") in {"progress", "metrics", "training_step"} or any(key in entry for key in ("step", "current", "percent")):
                job["progress"] = entry
            if entry.get("type") == "result":
                job["result"] = entry.get("result", entry.get("data", entry))
            if entry.get("type") == "error":
                job["error"] = str(entry.get("message", entry.get("error", entry)))
            with (self.store / job_id / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def list(self) -> list[dict]:
        self.active()
        with self._lock:
            return [dict(job) for job in sorted(self._jobs.values(), key=lambda item: item["started_at"], reverse=True)]

    def get(self, job_id: str, after: int = 0) -> dict:
        with self._lock:
            if job_id not in self._jobs:
                raise HTTPException(404, "Execução não encontrada")
            events = list(self._events.get(job_id, []))
            if not events:
                path = self.store / job_id / "events.jsonl"
                if path.is_file():
                    with path.open(encoding="utf-8") as stream:
                        lines = deque(stream, maxlen=1500)
                    events = [json.loads(line) for line in lines if line.strip()]
            return {"job": dict(self._jobs[job_id]), "events": [event for event in events if event["sequence"] > after], "cursor": events[-1]["sequence"] if events else after}

    def start(self, kind: str, config: dict) -> dict:
        if kind not in {"pairs", "score", "preprocess", "train", "model_setup"}:
            raise HTTPException(400, "Operação desconhecida")
        with self._lock:
            if self.active():
                raise HTTPException(409, "Já existe uma execução MR-FlowDPO ativa")
            if self.task_manager and self.task_manager.active_operation():
                raise HTTPException(409, "Outra operação está ativa na interface SFT/datasets")
            if kind == "train":
                try:
                    config = _config_module().validate_config(config)
                except (ValueError, OSError) as exc:
                    raise HTTPException(422, str(exc)) from exc
            if not isinstance(config.get("options", {}), dict):
                raise HTTPException(422, "Opções precisam ser um objeto JSON")
            job_id = uuid.uuid4().hex[:12]
            directory = self.store / job_id
            directory.mkdir(parents=True, exist_ok=False)
            config_path, stop_path = directory / "config.json", directory / "stop.request"
            _write_json(config_path, config)
            python = self.root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            executable = str(python) if python.is_file() else sys.executable
            command = [executable, "-u", str(self.root / "jk_step.py")]
            options = json.dumps(config.get("options", {}), ensure_ascii=False)
            if kind == "train":
                command += ["train", "--config", str(config_path), "--stop-file", str(stop_path)]
            elif kind == "model_setup":
                command += ["models", "setup", "--checkpoint-dir", _required(config, "checkpoint_dir")]
                if config.get("weights_only", False):
                    command += ["--weights-only"]
            elif kind == "score":
                command += ["pairs", "score", "--input", _required(config, "input_manifest_or_audio_dir"), "--output", _required(config, "output_file"), "--options", options, "--stop-file", str(stop_path)]
                if config.get("template", False):
                    command += ["--template"]
            elif kind == "pairs":
                command += ["pairs", "build", "--input", _required(config, "input_manifest_or_audio_dir"), "--output", _required(config, "output_dir"), "--options", options, "--stop-file", str(stop_path)]
            else:
                command += ["pairs", "preprocess", "--manifest", _required(config, "manifest"), "--checkpoint-dir", _required(config, "checkpoint_dir"), "--model-variant", _required(config, "model_variant"), "--output", _required(config, "output_dir"), "--options", options, "--stop-file", str(stop_path)]
            job = {"id": job_id, "kind": kind, "status": "running", "started_at": time.time(), "config": _jsonable(config), "config_path": str(config_path), "stop_path": str(stop_path), "progress": {}, "result": None}
            try:
                kwargs = {"cwd": str(self.root), "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True, "encoding": "utf-8", "errors": "replace", "env": {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}}
                if os.name == "nt":
                    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                process = subprocess.Popen(command, **kwargs)
            except OSError as exc:
                raise HTTPException(500, f"Não foi possível iniciar: {exc}") from exc
            job["pid"] = process.pid
            self._jobs[job_id], self._processes[job_id] = job, process
            self._persist(job)
            threading.Thread(target=self._watch, args=(job_id, process), daemon=True).start()
            return dict(job)

    def _watch(self, job_id: str, process: subprocess.Popen) -> None:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        event = {"type": "log", "message": line}
                except ValueError:
                    event = {"type": "log", "message": line}
                self._emit(job_id, event)
            exit_code = process.wait()
            with self._lock:
                job = self._jobs[job_id]
                job["exit_code"] = exit_code
                job["finished_at"] = time.time()
                job["status"] = "cancelled" if job["status"] == "stopping" else ("completed" if exit_code == 0 else "failed")
                if exit_code and not job.get("error") and job["status"] == "failed":
                    job["error"] = f"O processo terminou com código {exit_code}. Consulte o log."
                self._persist(job)
                self._processes.pop(job_id, None)
            self._emit(job_id, {"type": "status", "status": job["status"]})
        finally:
            if process.stdout:
                process.stdout.close()

    def stop(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self._jobs:
                raise HTTPException(404, "Execução não encontrada")
            job = self._jobs[job_id]
            if job["status"] == "orphaned":
                raise HTTPException(409, "O servidor reiniciou enquanto o processo continuou ativo. Encerre esse processo antes de iniciar outra execução.")
            if job["status"] not in {"running", "stopping"}:
                return dict(job)
            job["status"] = "stopping"
            Path(job["stop_path"]).touch()
            self._persist(job)
            if job["kind"] == "model_setup":
                process = self._processes[job_id]
                try:
                    if os.name == "nt":
                        # Windows venv python.exe can be a launcher with a real
                        # Python child. Target only this owned job's process tree.
                        result = subprocess.run(
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                            capture_output=True, text=True, errors="replace", timeout=10)
                        if result.returncode and process.poll() is None:
                            raise OSError(result.stderr.strip() or result.stdout.strip() or "taskkill failed")
                    else:
                        process.terminate()
                except (OSError, subprocess.TimeoutExpired) as exc:
                    job["status"] = "running"
                    self._persist(job)
                    raise HTTPException(500, f"Não foi possível interromper o download: {exc}") from exc
            self._emit(job_id, {"type": "log", "message": "Parada solicitada. O treinamento salva o estado antes de sair." if job["kind"] == "train" else "Download interrompido." if job["kind"] == "model_setup" else "Parada solicitada; a operação termina o item atual antes de sair."})
            return dict(job)


def _required(config: dict, field: str) -> str:
    value = str(config.get(field, "")).strip()
    if not value:
        raise HTTPException(422, f"Campo obrigatório: {field}")
    return value


def _config_module():
    return importlib.import_module("jk_step.config")


def _dataset_module():
    return importlib.import_module("jk_step.pairs")


def create_router(task_manager=None, token: str = "") -> tuple[APIRouter, JobManager]:
    router = APIRouter()
    manager = JobManager(task_manager)

    @router.get("/quality", response_class=HTMLResponse)
    async def quality():
        html = (_ROOT / "frontend" / "quality.html").read_text("utf-8")
        safe_token = json.dumps(token).replace("<", "\\u003c")
        html = html.replace("<body>", "<body>\n<script>window.__SIDESTEP_TOKEN__=" + safe_token + ";</script>", 1)
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    @router.get("/api/mrflow/schema")
    async def schema():
        config = _config_module()
        return {"defaults": _jsonable(config.schema_defaults()), "fields": _jsonable(config.schema_fields())}

    @router.get("/api/mrflow/presets")
    async def presets():
        module = importlib.import_module("jk_step.cli")
        builtin = _jsonable(module.get_presets())
        saved = []
        directory = _STORE / "presets"
        if directory.is_dir():
            for path in sorted(directory.glob("*.json")):
                try:
                    saved.append(json.loads(path.read_text("utf-8")))
                except (OSError, ValueError):
                    continue
        return {"builtin": builtin, "saved": saved}

    @router.post("/api/mrflow/presets")
    async def save_preset(body: dict):
        name = _required(body, "name")
        if len(name) > 100:
            raise HTTPException(422, "Use um nome de até 100 caracteres")
        value = {"name": name, "config": body.get("config", {})}
        if not isinstance(value["config"], dict):
            raise HTTPException(422, "A configuração precisa ser um objeto JSON")
        import hashlib
        _write_json(_STORE / "presets" / (hashlib.sha256(name.encode()).hexdigest()[:20] + ".json"), value)
        return value

    @router.get("/api/mrflow/jobs")
    async def jobs():
        return {"jobs": manager.list(), "active": manager.active()}

    @router.post("/api/mrflow/jobs")
    async def start_job(body: dict):
        config = body.get("config", {})
        if not isinstance(config, dict):
            raise HTTPException(422, "Configuração deve ser um objeto JSON")
        return manager.start(_required(body, "kind"), config)

    @router.get("/api/mrflow/jobs/{job_id}")
    async def get_job(job_id: str, after: int = 0):
        return manager.get(job_id, max(0, after))

    @router.post("/api/mrflow/jobs/{job_id}/stop")
    async def stop_job(job_id: str):
        return manager.stop(job_id)

    @router.post("/api/mrflow/pairs/validate")
    async def validate(body: dict):
        try:
            result = await asyncio.to_thread(_dataset_module().validate_pairs, _required(body, "manifest"))
            return _jsonable(result)
        except (OSError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/api/mrflow/scores/read")
    async def read_scores(body: dict):
        path = Path(_required(body, "path")).expanduser()
        if not path.is_absolute():
            path = _ROOT / path
        try:
            if path.suffix.lower() != ".json":
                raise ValueError("Use um arquivo JSON de candidatos/avaliações")
            if path.stat().st_size > 20 * 1024 * 1024:
                raise ValueError("O editor aceita JSON até 20 MB. Divida os candidatos ou edite o arquivo externamente.")
            data = json.loads(path.read_text("utf-8-sig"))
            if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
                raise ValueError("Arquivo precisa conter um objeto com lista samples")
            return {"path": str(path.resolve()), "data": data}
        except (OSError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/api/mrflow/training/resume-config")
    async def resume_config(body: dict):
        directory = Path(_required(body, "adapter_dir")).expanduser()
        if not directory.is_absolute():
            directory = _ROOT / directory
        try:
            if not (directory / "training_state.pt").is_file():
                raise ValueError("Esse diretório não contém training_state.pt; use init_adapter para um LoRA sem estado de retomada.")
            config = json.loads((directory / "training_config.json").read_text("utf-8-sig"))
            if not isinstance(config, dict):
                raise ValueError("training_config.json precisa conter um objeto JSON")
            config.update(resume_from=str(directory.resolve()), init_adapter="")
            return {"config": _config_module().validate_config(config)}
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/api/mrflow/scores/save")
    async def save_scores(body: dict):
        path = Path(_required(body, "path")).expanduser()
        if not path.is_absolute():
            path = _ROOT / path
        data = body.get("data")
        if path.suffix.lower() != ".json" or not isinstance(data, dict) or not isinstance(data.get("samples"), list):
            raise HTTPException(422, "Use um arquivo .json com lista samples")
        try:
            json.dumps(data, allow_nan=False)
            _write_json(path, data)
            return {"path": str(path.resolve()), "samples": len(data["samples"])}
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/api/mrflow/pairs/preview")
    async def preview(body: dict):
        path = Path(_required(body, "manifest")).expanduser()
        if not path.is_absolute():
            path = _ROOT / path
        if not path.is_file():
            raise HTTPException(404, "Manifesto não encontrado")
        try:
            maximum = min(20, max(1, int(body.get("limit", 5))))
            payload, _ = _dataset_module().read_manifest(path)
            rows = payload.get("pairs", payload.get("samples", []))
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ValueError("Manifesto precisa conter uma lista de pares")
            return {"path": str(path.resolve()), "count": len(rows), "pairs": rows[:maximum]}
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/api/mrflow/pairs/audio")
    async def pair_audio(manifest: str, index: int, branch: str):
        if branch not in {"chosen", "rejected"} or index < 0:
            raise HTTPException(422, "Par ou ramo inválido")
        try:
            dataset, base = _dataset_module().read_manifest(manifest)
            pairs = dataset.get("pairs", [])
            if index >= len(pairs):
                raise HTTPException(404, "Par não encontrado")
            candidate = Path(pairs[index][branch]).expanduser()
            path = (candidate if candidate.is_absolute() else base / candidate).resolve()
            if not path.is_file() or path.suffix.lower() not in _dataset_module().AUDIO_EXTENSIONS:
                raise HTTPException(404, "Áudio não encontrado")
            return FileResponse(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    return router, manager


def install(app, task_manager=None, token: str = "") -> JobManager:
    """Add routes and serialize starts across MRFlow and existing modules."""
    router, manager = create_router(task_manager, token)
    app.include_router(router)
    app.state.mrflow_manager = manager
    launch_lock = asyncio.Lock()
    old_starters = {"/api/train/start", "/api/preprocess/start", "/api/ppplus/start", "/api/captions/start", "/api/audio-analyze/start", "/api/audio-analyze/one"}

    @app.middleware("http")
    async def operation_mutex(request: Request, call_next):
        path = request.url.path
        if request.method == "POST" and (path in old_starters or path == "/api/mrflow/jobs"):
            from jk_engine.gui.security import _extract_token
            if _extract_token(request.scope) != token:
                return await call_next(request)
            async with launch_lock:
                if path in old_starters and manager.active():
                    from fastapi.responses import JSONResponse
                    return JSONResponse({"error": "Uma execução MR-FlowDPO está ativa. Pare ou aguarde a execução atual."}, status_code=409)
                return await call_next(request)
        return await call_next(request)
    return manager
