"""Authenticated UI/CLI integration checks without GPU, models or downloads."""
import io
import json
import math
import time
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from jk_step import web
from jk_engine.gui.server import create_app


TOKEN = "jk-web-test-token"
HEADERS = {"Authorization": "Bearer " + TOKEN}


@pytest.fixture
def ui(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "_STORE", tmp_path / "state")
    app = create_app(token=TOKEN, port=8771)
    manager = app.app.state.mrflow_manager
    manager.store = tmp_path / "jobs"
    manager._jobs.clear()
    with TestClient(app, base_url="http://127.0.0.1:8771") as client:
        yield client, manager


def audio_pair(tmp_path):
    paths = []
    for name in ("chosen", "rejected"):
        path = tmp_path / (name + ".wav")
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(2)
            audio.setsampwidth(2)
            audio.setframerate(48000)
            audio.writeframes(b"\0" * (4800 * 4))
        paths.append(path)
    manifest = tmp_path / "pairs.json"
    manifest.write_text(json.dumps({"version": 1, "pairs": [{
        "id": "pair", "group_id": "song", "split": "train",
        "chosen": str(paths[0]), "rejected": str(paths[1]),
        "caption": "Piano and voice", "lyrics": "One two three"}]}), "utf-8")
    return manifest, paths


def wait_job(manager, job_id):
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        report = manager.get(job_id)
        if report["job"]["status"] not in {"running", "stopping"}:
            return report
        time.sleep(0.03)
    pytest.fail("CLI job did not terminate within 12 seconds")


def test_existing_security_and_all_schema_fields(ui):
    client, _ = ui
    assert client.get("/quality").status_code == 401
    assert client.get("/api/mrflow/schema").status_code == 401
    page = client.get("/quality", params={"token": TOKEN})
    assert page.status_code == 200
    assert TOKEN in page.text and "JK-Step" in page.text
    assert client.get("/", headers=HEADERS).status_code == 200
    schema = client.get("/api/mrflow/schema", headers=HEADERS).json()
    assert {field["name"] for field in schema["fields"]} == set(schema["defaults"])
    assert schema["defaults"]["rank"] == 32
    assert {"checkpoint_file", "model_config_dir", "resume_from", "rank_pattern",
            "multi_gpu", "gpu_ids", "distributed_backend"} <= set(schema["defaults"])
    assert schema["defaults"]["multi_gpu"] is False
    assert schema["defaults"]["gpu_ids"] == ["0", "1"]
    assert client.get("/api/mrflow/presets", headers=HEADERS).status_code == 200
    assert client.get("/api/mrflow/schema", headers={**HEADERS, "Host": "untrusted.example"}).status_code == 403
    assert client.post("/api/mrflow/jobs", headers=HEADERS,
                       json={"kind": "train", "config": {}}).status_code == 422
    assert client.post("/api/mrflow/jobs", headers=HEADERS,
                       json={"kind": "unknown"}).status_code == 400


def test_pair_validation_preview_and_authenticated_audio(ui, tmp_path):
    client, _ = ui
    manifest, _ = audio_pair(tmp_path)
    body = {"manifest": str(manifest)}
    validation = client.post("/api/mrflow/pairs/validate", headers=HEADERS, json=body)
    assert validation.status_code == 200 and validation.json()["valid"]
    preview = client.post("/api/mrflow/pairs/preview", headers=HEADERS, json=body)
    assert preview.status_code == 200 and preview.json()["count"] == 1
    params = {"manifest": str(manifest), "index": 0, "branch": "chosen"}
    assert client.get("/api/mrflow/pairs/audio", params=params).status_code == 401
    audio = client.get("/api/mrflow/pairs/audio", params=params, headers=HEADERS)
    assert audio.status_code == 200 and audio.content.startswith(b"RIFF")
    assert client.get("/api/mrflow/pairs/audio", params={**params, "branch": "invalid"}, headers=HEADERS).status_code == 422
    assert client.post("/api/mrflow/pairs/preview", headers=HEADERS,
                       json={**body, "limit": "wrong"}).status_code == 422
    invalid = tmp_path / "invalid.json"
    invalid.write_text("123", "utf-8")
    assert client.post("/api/mrflow/pairs/preview", headers=HEADERS,
                       json={"manifest": str(invalid)}).status_code == 422


def test_manual_score_editor_round_trip_and_validation(ui, tmp_path):
    client, _ = ui
    output = tmp_path / "scores.json"
    data = {"samples": [{"audio_path": "chosen.wav", "group_id": "song",
                          "scores": {"vocal_clarity": 0.8, "lyric_fidelity": 1}}]}
    save = client.post("/api/mrflow/scores/save", headers=HEADERS,
                       json={"path": str(output), "data": data})
    assert save.status_code == 200 and save.json()["samples"] == 1
    read = client.post("/api/mrflow/scores/read", headers=HEADERS, json={"path": str(output)})
    assert read.status_code == 200 and read.json()["data"] == data
    assert client.post("/api/mrflow/scores/save", headers=HEADERS,
                       json={"path": str(output), "data": {"pairs": []}}).status_code == 422
    assert client.post("/api/mrflow/scores/save", headers=HEADERS,
                       json={"path": str(tmp_path / "scores.txt"), "data": data}).status_code == 422
    assert client.post("/api/mrflow/scores/read", headers=HEADERS,
                       json={"path": str(tmp_path / "absent.json")}).status_code == 422
    # Scientific packages sometimes print non-finite diagnostics. They must
    # remain renderable JSON rather than breaking every job-poll response.
    assert web._jsonable({"value": math.nan}) == {"value": "NaN"}


def test_mrflow_and_existing_toolkit_share_operation_mutex(ui):
    client, manager = ui
    manager._jobs["active"] = {"id": "active", "kind": "train", "status": "running", "started_at": 1}
    assert client.post("/api/train/start", headers=HEADERS, json={}).status_code == 409
    assert client.post("/api/train/start", json={}).status_code == 401
    assert client.post("/api/mrflow/jobs", headers=HEADERS,
                       json={"kind": "pairs", "config": {}}).status_code == 409


def test_resume_restores_saved_configuration_without_loading_tensors(ui, tmp_path):
    client, _ = ui
    checkpoint = tmp_path / "checkpoint-00000002"
    checkpoint.mkdir()
    (checkpoint / "training_state.pt").write_bytes(b"not-deserialized-by-web")
    saved = {"pairs_manifest": "pairs.json", "rank": 64, "alpha": 128,
             "learning_rate": 0.000001, "init_adapter": "old-initial-adapter"}
    (checkpoint / "training_config.json").write_text(json.dumps(saved), "utf-8")
    response = client.post("/api/mrflow/training/resume-config", headers=HEADERS,
                           json={"adapter_dir": str(checkpoint)})
    assert response.status_code == 200
    config = response.json()["config"]
    assert config["rank"] == 64 and config["alpha"] == 128
    assert config["learning_rate"] == saved["learning_rate"]
    assert config["resume_from"] == str(checkpoint.resolve()) and config["init_adapter"] == ""
    assert client.post("/api/mrflow/training/resume-config", headers=HEADERS,
                       json={"adapter_dir": str(tmp_path / "absent")}).status_code == 422


def test_real_pair_build_cli_progress_result_and_cursor(ui, tmp_path):
    client, manager = ui
    source, _ = audio_pair(tmp_path)
    job = client.post("/api/mrflow/jobs", headers=HEADERS, json={"kind": "pairs", "config": {
        "input_manifest_or_audio_dir": str(source), "output_dir": str(tmp_path / "output"),
        "options": {"mode": "import", "validation_fraction": 0}}}).json()
    report = wait_job(manager, job["id"])
    assert report["job"]["status"] == "completed", report
    assert report["job"]["result"]["pairs"] == 1
    assert Path(report["job"]["result"]["manifest"]).is_file()
    assert any(event["type"] == "result" for event in report["events"])
    assert manager.get(job["id"], after=report["cursor"])["events"] == []


def test_manual_score_template_real_cli_needs_no_models(ui, tmp_path):
    client, manager = ui
    _, paths = audio_pair(tmp_path)
    source = tmp_path / "candidates.json"
    source.write_text(json.dumps({"samples": [{"audio_path": str(path), "group_id": "song",
        "caption": "Piano and voice", "lyrics": "One two three"} for path in paths]}), "utf-8")
    output = tmp_path / "template.json"
    job = client.post("/api/mrflow/jobs", headers=HEADERS, json={"kind": "score", "config": {
        "input_manifest_or_audio_dir": str(source), "output_file": str(output), "template": True,
        "options": {"axes": ["vocal_clarity", "lyric_fidelity"]}}}).json()
    report = wait_job(manager, job["id"])
    assert report["job"]["status"] == "completed", report
    assert report["job"]["result"]["samples"] == 2
    document = json.loads(output.read_text("utf-8"))
    assert all(sample["scores"] == {"vocal_clarity": None, "lyric_fidelity": None} for sample in document["samples"])


def test_model_setup_uses_cli_contract_without_downloading(ui, monkeypatch, tmp_path):
    _, manager = ui
    commands = []
    class FakeProcess:
        pid = 123456789
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.stdout = io.StringIO('{"event":"result","result":{"checkpoint_dir":"checkpoints"}}\n')
        def wait(self):
            return 0
    monkeypatch.setattr(web.subprocess, "Popen", FakeProcess)
    job = manager.start("model_setup", {"checkpoint_dir": str(tmp_path / "checkpoints"), "weights_only": True})
    report = wait_job(manager, job["id"])
    assert report["job"]["status"] == "completed"
    assert commands[0][-5:] == ["models", "setup", "--checkpoint-dir", str(tmp_path / "checkpoints"), "--weights-only"]


@pytest.mark.parametrize("platform,kind", [("nt", "model_setup"), ("posix", "model_setup"), ("nt", "train")])
def test_cancel_owned_download_tree_and_keep_training_cooperative(ui, monkeypatch, platform, kind):
    _, manager = ui
    directory = manager.store / "owned-job"
    directory.mkdir(parents=True)
    stop_path = directory / "stop.request"
    process = Mock(pid=123456789)
    process.poll.return_value = None
    manager._jobs["owned-job"] = {"id": "owned-job", "kind": kind, "status": "running",
                                  "started_at": 1, "stop_path": str(stop_path)}
    manager._processes["owned-job"] = process
    run = Mock(return_value=SimpleNamespace(returncode=0))
    # Replace only web.py's os binding, without changing global pathlib state.
    monkeypatch.setattr(web, "os", SimpleNamespace(name=platform))
    monkeypatch.setattr(web.subprocess, "run", run)
    result = manager.stop("owned-job")
    assert result["status"] == "stopping" and stop_path.is_file()
    if kind == "train":
        run.assert_not_called()
        process.terminate.assert_not_called()
    elif platform == "nt":
        assert run.call_args.args[0] == ["taskkill", "/PID", "123456789", "/T", "/F"]
        assert run.call_args.kwargs["creationflags"] == getattr(web.subprocess, "CREATE_NO_WINDOW", 0)
        process.terminate.assert_not_called()
    else:
        run.assert_not_called()
        process.terminate.assert_called_once_with()


def test_failed_download_tree_cancel_keeps_owned_job_active(ui, monkeypatch):
    _, manager = ui
    directory = manager.store / "owned-job"
    directory.mkdir(parents=True)
    process = Mock(pid=123456789)
    process.poll.return_value = None
    manager._jobs["owned-job"] = {"id": "owned-job", "kind": "model_setup", "status": "running",
                                  "started_at": 1, "stop_path": str(directory / "stop.request")}
    manager._processes["owned-job"] = process
    monkeypatch.setattr(web, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(web.subprocess, "run", Mock(return_value=SimpleNamespace(
        returncode=1, stdout="", stderr="Permission denied")))
    with pytest.raises(HTTPException) as error:
        manager.stop("owned-job")
    assert error.value.status_code == 500
    assert manager._jobs["owned-job"]["status"] == "running"
