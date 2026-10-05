"""Synchronous LoRA data parallelism and portable multi-GPU process launcher.

Only adapter gradients are synchronized. Frozen ACE weights and the disabled-
adapter reference stay local, with no second XL copy. Windows uses a local
shared-memory broker because some official PyTorch wheels advertise Gloo while
shipping no usable Gloo transport, and NCCL is unavailable there.
"""
from __future__ import annotations

import datetime
import atexit
import json
import multiprocessing.managers
import multiprocessing.shared_memory
import os
from pathlib import Path
import queue
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any


class _CollectiveBroker:
    """Single-host rendezvous; large tensors never travel through the socket."""
    def __init__(self, world_size: int, timeout: float = 600):
        self.world_size = world_size
        self.timeout = timeout
        self.condition = threading.Condition()
        self.slots = {}

    def _reduce_arrays(self, values, operation):
        import numpy as np
        handles, arrays = [], []
        try:
            descriptors = [values[rank] for rank in range(self.world_size)]
            first = descriptors[0]
            for descriptor in descriptors:
                if descriptor["shape"] != first["shape"] or descriptor["dtype"] != first["dtype"]:
                    raise RuntimeError("Collective tensor shapes or dtypes differ between ranks")
                handle = multiprocessing.shared_memory.SharedMemory(name=descriptor["name"])
                handles.append(handle)
                arrays.append(np.ndarray(tuple(descriptor["shape"]),dtype=np.dtype(descriptor["dtype"]),buffer=handle.buf))
            result = arrays[0].copy()
            if operation == "tensor_sum":
                for array in arrays[1:]:
                    result += array
            for array in arrays:
                np.copyto(array,result)
            return True
        finally:
            arrays.clear()
            # The loop variable also holds a view of the final segment.
            if "array" in locals():
                del array
            for handle in handles:
                handle.close()

    def collective(self, sequence: int, rank: int, operation: str, value: Any):
        if not 0 <= rank < self.world_size:
            raise ValueError("Invalid distributed rank")
        deadline = time.monotonic() + self.timeout
        with self.condition:
            slot = self.slots.setdefault(sequence,{"operation":operation,"values":{},"ready":False,
                                                  "result":None,"error":None,"delivered":set()})
            if slot["operation"] != operation or rank in slot["values"]:
                slot["error"] = "Collective order differs between workers"
                slot["ready"] = True
                self.condition.notify_all()
            else:
                slot["values"][rank] = value
                if len(slot["values"]) == self.world_size and not slot["ready"]:
                    values = slot["values"]
                    try:
                        if operation in ("tensor_sum","tensor_broadcast"):
                            slot["result"] = self._reduce_arrays(values,operation)
                        elif operation == "barrier":
                            slot["result"] = None
                        elif operation == "any":
                            slot["result"] = any(values.values())
                        elif operation == "sum_dict":
                            keys = sorted(values[0])
                            if any(sorted(item) != keys for item in values.values()):
                                raise RuntimeError("Collective metric keys differ between ranks")
                            slot["result"] = {key:sum(values[r][key] for r in range(self.world_size)) for key in keys}
                        elif operation == "gather_object":
                            slot["result"] = [values[r] for r in range(self.world_size)]
                        elif operation == "broadcast_object":
                            slot["result"] = values[0]
                        else:
                            raise ValueError(f"Unknown local collective: {operation}")
                    except Exception as error:
                        slot["error"] = f"{type(error).__name__}: {error}"
                    slot["ready"] = True
                    self.condition.notify_all()
            while not slot["ready"]:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    slot["error"] = f"Local collective timed out waiting for all {self.world_size} ranks"
                    slot["ready"] = True
                    self.condition.notify_all()
                    break
                self.condition.wait(timeout=remaining)
            result, error = slot["result"], slot["error"]
            slot["delivered"].add(rank)
            if len(slot["delivered"]) == self.world_size:
                del self.slots[sequence]
            if error:
                raise RuntimeError(error)
            return result


_SERVER_BROKER = None


def _initialize_local_server(world_size, timeout):
    global _SERVER_BROKER
    _SERVER_BROKER = _CollectiveBroker(world_size,timeout)


def _get_local_server():
    return _SERVER_BROKER


class _LocalManager(multiprocessing.managers.BaseManager):
    pass


_LocalManager.register("get_broker",callable=_get_local_server,exposed=("collective",))


class LocalBrokerHandle:
    """Parent-owned broker. Bind only loopback; authenticate every connection."""
    def __init__(self, world_size: int, timeout: float = 600):
        if world_size < 2:
            raise ValueError("The local broker requires at least two workers")
        self.authkey = secrets.token_bytes(32)
        self.manager = _LocalManager(address=("127.0.0.1",0),authkey=self.authkey,
                                    ctx=__import__("multiprocessing").get_context("spawn"))
        self.manager.start(initializer=_initialize_local_server,initargs=(world_size,timeout))
        self.address = self.manager.address
        self.closed = False

    def worker_environment(self) -> dict[str,str]:
        return {"JK_STEP_LOCAL_ADDRESS":json.dumps(self.address),
                "JK_STEP_LOCAL_AUTH":self.authkey.hex()}

    def close(self):
        if not self.closed:
            self.manager.shutdown()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()


def start_local_broker(world_size: int, timeout: float = 600) -> LocalBrokerHandle:
    """Start a broker before spawning workers; also useful for CPU integration tests."""
    return LocalBrokerHandle(world_size,timeout)


class _LocalClient:
    def __init__(self, rank: int, world_size: int):
        address = json.loads(os.environ["JK_STEP_LOCAL_ADDRESS"])
        self.manager = _LocalManager(address=(str(address[0]),int(address[1])),
                                    authkey=bytes.fromhex(os.environ["JK_STEP_LOCAL_AUTH"]))
        self.manager.connect()
        self.proxy = self.manager.get_broker()
        self.rank, self.world_size, self.sequence = rank, world_size, 0
        self.segment = None
        atexit.register(self.close)

    def collective(self, operation, value=None):
        sequence = self.sequence
        self.sequence += 1
        return self.proxy.collective(sequence,self.rank,operation,value)

    def tensor_collective(self, tensor, operation):
        import numpy as np
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError("Local collectives require contiguous CPU tensors")
        source = tensor.numpy()
        required = max(1,source.nbytes)
        if self.segment is None or self.segment.size < required:
            self.close()
            self.segment = multiprocessing.shared_memory.SharedMemory(create=True,size=required)
        target = np.ndarray(source.shape,dtype=source.dtype,buffer=self.segment.buf)
        np.copyto(target,source)
        self.collective(operation,{"name":self.segment.name,"shape":list(source.shape),"dtype":source.dtype.str})
        np.copyto(source,target)

    def close(self):
        if self.segment is not None:
            self.segment.close()
            self.segment.unlink()
            self.segment = None


_LOCAL_CLIENTS = {}


@dataclass
class DistributedContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str = "none"
    device: Any = None
    owns_group: bool = False
    local_client: Any = None

    @property
    def active(self):
        return self.world_size > 1

    @property
    def is_main(self):
        return self.rank == 0

    def _collective_device(self):
        return self.device if self.backend == "nccl" else "cpu"

    def barrier(self):
        if self.active:
            if self.backend == "local":
                self.local_client.collective("barrier")
                return
            import torch.distributed as dist
            dist.barrier()

    def any(self, flag: bool) -> bool:
        if not self.active:
            return bool(flag)
        if self.backend == "local":
            return bool(self.local_client.collective("any",bool(flag)))
        import torch
        import torch.distributed as dist
        value = torch.tensor(int(flag),device=self._collective_device(),dtype=torch.int32)
        dist.all_reduce(value,op=dist.ReduceOp.MAX)
        return bool(value.item())

    def sum(self, value: float) -> float:
        return self.sum_dict({"value":value})["value"]

    def sum_dict(self, values: dict[str,float]) -> dict[str,float]:
        if not self.active:
            return values
        if self.backend == "local":
            return self.local_client.collective("sum_dict",values)
        import torch
        import torch.distributed as dist
        keys = sorted(values)
        tensor = torch.tensor([values[key] for key in keys],device=self._collective_device(),dtype=torch.float64)
        dist.all_reduce(tensor,op=dist.ReduceOp.SUM)
        return dict(zip(keys,tensor.cpu().tolist()))

    def gather_object(self, value: Any) -> list[Any]:
        if not self.active:
            return [value]
        if self.backend == "local":
            return self.local_client.collective("gather_object",value)
        import torch.distributed as dist
        values = [None] * self.world_size
        dist.all_gather_object(values,value)
        return values

    def broadcast_object(self, value: Any = None) -> Any:
        if not self.active:
            return value
        if self.backend == "local":
            return self.local_client.collective("broadcast_object",value if self.is_main else None)
        import torch.distributed as dist
        container = [value if self.is_main else None]
        dist.broadcast_object_list(container,src=0)
        return container[0]

    def synchronize_gradients(self, parameters: list[Any], *, average: bool = False,
                              bucket_bytes: int = 16*1024*1024) -> None:
        """Synchronize dense trainable gradients after an accumulation window.

        The engine scales each microbatch by its fraction of the GLOBAL weight
        mass before calling this SUM reduction. This handles uneven pair weights
        and final microbatches correctly. Gloo stages CPU float32 buckets rather
        than attempting unsupported Windows CUDA collectives.
        """
        if not self.active:
            return
        import torch
        import torch.distributed as dist
        presence = torch.tensor([int(p.grad is not None) for p in parameters],dtype=torch.int32,
                                device=self._collective_device())
        if self.backend == "local":
            self.local_client.tensor_collective(presence,"tensor_sum")
        else:
            dist.all_reduce(presence,op=dist.ReduceOp.SUM)
        active = [p for p,has_gradient in zip(parameters,presence.cpu().tolist()) if has_gradient]
        buckets, current, size = [], [], 0
        limit = max(1,bucket_bytes//4)
        for parameter in active:
            if current and size + parameter.numel() > limit:
                buckets.append(current);current=[];size=0
            current.append(parameter);size += parameter.numel()
        if current:
            buckets.append(current)
        for bucket in buckets:
            total = sum(p.numel() for p in bucket)
            cpu_staging = self.backend != "nccl"
            buffer = torch.empty(total,dtype=torch.float32,device="cpu" if cpu_staging else self.device,
                                 pin_memory=cpu_staging and self.device.type=="cuda")
            offset = 0
            for parameter in bucket:
                target = buffer[offset:offset+parameter.numel()]
                if parameter.grad is None:
                    target.zero_()
                else:
                    if parameter.grad.is_sparse:
                        raise ValueError("Sparse adapter gradients are not supported")
                    target.copy_(parameter.grad.detach().reshape(-1),non_blocking=cpu_staging and self.device.type=="cuda")
                offset += parameter.numel()

            if cpu_staging and self.device.type=="cuda":
                torch.cuda.synchronize(self.device)
            if self.backend == "local":
                self.local_client.tensor_collective(buffer,"tensor_sum")
            else:
                dist.all_reduce(buffer,op=dist.ReduceOp.SUM)
            if average:
                buffer.div_(self.world_size)
            reduced = buffer.to(self.device,non_blocking=cpu_staging and self.device.type=="cuda")
            offset = 0
            for parameter in bucket:
                value = reduced[offset:offset+parameter.numel()].view_as(parameter)
                if parameter.grad is None:
                    parameter.grad = value.to(dtype=parameter.dtype).clone()
                else:
                    parameter.grad.copy_(value)
                offset += parameter.numel()

    def synchronize_parameters(self, parameters: list[Any]) -> None:
        """Broadcast only LoRA parameters so every worker starts identically."""
        if not self.active:
            return
        import torch
        import torch.distributed as dist
        for parameter in parameters:
            value = parameter.detach().to(self._collective_device(),dtype=torch.float32).contiguous()
            if self.backend == "local":
                self.local_client.tensor_collective(value,"tensor_broadcast")
            else:
                dist.broadcast(value,src=0)
            parameter.data.copy_(value.to(parameter.device,dtype=parameter.dtype))

    def close(self):
        if self.owns_group:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
            self.owns_group = False


def initialize_distributed(config: dict, device: Any) -> DistributedContext:
    """Initialize the worker rank; direct single-GPU calls remain unchanged."""
    import torch.distributed as dist
    world = int(os.environ.get("WORLD_SIZE","1"))
    if world <= 1:
        return DistributedContext(device=device)
    backend = config.get("distributed_backend","auto")
    if backend == "auto":
        backend = "local" if sys.platform == "win32" else (
            "nccl" if device.type=="cuda" and dist.is_nccl_available() else "gloo")
    if backend == "local":
        if not os.environ.get("JK_STEP_LOCAL_ADDRESS") or not os.environ.get("JK_STEP_LOCAL_AUTH"):
            raise RuntimeError("Local parallelism needs the parent IPC broker: use launch_training, "
                               "or start_local_broker before spawning rank workers")
        rank = int(os.environ.get("RANK","0"))
        key = (os.environ["JK_STEP_LOCAL_ADDRESS"],os.environ["JK_STEP_LOCAL_AUTH"],rank,world)
        client = _LOCAL_CLIENTS.get(key)
        if client is None:
            client = _LocalClient(rank,world)
            _LOCAL_CLIENTS[key] = client
        return DistributedContext(rank=rank,world_size=world,local_rank=int(os.environ.get("LOCAL_RANK","0")),
                                  backend="local",device=device,local_client=client)
    if not dist.is_available():
        raise RuntimeError("This PyTorch installation has no distributed support; use backend local")
    if backend not in ("gloo","nccl"):
        raise ValueError("distributed_backend must be auto, local, gloo, or nccl")
    if backend == "nccl" and (device.type != "cuda" or not dist.is_nccl_available()):
        raise ValueError("NCCL requires a CUDA PyTorch build with NCCL; use Gloo on Windows")
    if backend == "gloo" and not dist.is_gloo_available():
        raise ValueError("This PyTorch build has no Gloo backend")
    owns = not dist.is_initialized()
    if owns:
        file_store = os.environ.get("JK_STEP_DIST_STORE")
        if file_store:
            store = dist.FileStore(str(Path(file_store).resolve()),world)
            try:
                dist.init_process_group(backend=backend,store=store,
                    rank=int(os.environ.get("RANK","0")),world_size=world,
                    timeout=datetime.timedelta(seconds=600))
            except RuntimeError as error:
                if backend=="gloo" and "unsupported gloo device" in str(error):
                    raise RuntimeError("This PyTorch wheel has no usable Gloo transport. "
                                       "Select distributed_backend='auto' or 'local' on Windows.") from error
                raise
        else:
            dist.init_process_group(backend=backend,init_method="env://",
                                    timeout=datetime.timedelta(seconds=600))
    return DistributedContext(rank=dist.get_rank(),world_size=dist.get_world_size(),
        local_rank=int(os.environ.get("LOCAL_RANK","0")),backend=dist.get_backend(),device=device,owns_group=owns)


class EvaluationShardSampler:
    """Exact validation sharding: no repeated examples to fill equal lengths."""
    def __init__(self,dataset,rank:int,world_size:int):
        self.length,self.rank,self.world_size=len(dataset),rank,world_size
    def __iter__(self):
        return iter(range(self.rank,self.length,self.world_size))
    def __len__(self):
        return len(range(self.rank,self.length,self.world_size))


def _terminate_process_tree(process):
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,check=False,creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        try:
            os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:
            pass


def _launch_windows_workers(config_path: Path, shared_stop: Path, result_path: Path,
                            store_path: Path, gpu_ids: list[str], env: dict,
                            progress, stop_event) -> dict:
    """Use native rank subprocesses with either local IPC or a Gloo FileStore."""
    processes, readers, lines, ended, tail = [], [], queue.Queue(), set(), []
    command = [sys.executable,"-m","jk_step.distributed","--config",str(config_path),
               "--stop-file",str(shared_stop),"--result",str(result_path)]
    def reader(rank,process):
        try:
            for line in process.stdout:
                lines.put((rank,line.rstrip()))
        finally:
            lines.put((rank,None))
    def publish(rank,line):
        nonlocal tail
        if line is None:
            ended.add(rank)
            return
        tail.append(f"[rank {rank}] {line}");tail=tail[-100:]
        try:
            record=json.loads(line)
        except json.JSONDecodeError:
            record={"event":"distributed_log","rank":rank,"message":line}
        if progress:
            progress(record)
    try:
        for rank in range(len(gpu_ids)):
            child_env = {**env,"RANK":str(rank),"LOCAL_RANK":str(rank),
                         "WORLD_SIZE":str(len(gpu_ids)),"JK_STEP_DIST_STORE":str(store_path)}
            process = subprocess.Popen(command,cwd=str(Path(__file__).resolve().parent.parent),env=child_env,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=="win32" else 0,
                start_new_session=sys.platform!="win32")
            processes.append(process)
            thread = threading.Thread(target=reader,args=(rank,process),daemon=True)
            readers.append(thread)
            thread.start()
        stopped_at = None
        while len(ended)<len(processes) or any(process.poll() is None for process in processes):
            try:
                if (stop_event is not None and stop_event.is_set()) or shared_stop.exists():
                    shared_stop.parent.mkdir(parents=True,exist_ok=True)
                    shared_stop.touch()
                    stopped_at = stopped_at or time.monotonic()
                failures = [(rank,process.returncode) for rank,process in enumerate(processes)
                            if process.poll() is not None and process.returncode != 0]
                if failures:
                    # Preserve the actual exception even if process exit was
                    # observed before its final stdout lines were consumed.
                    for rank,_ in failures:
                        readers[rank].join(timeout=.2)
                    while True:
                        try:
                            publish(*lines.get_nowait())
                        except queue.Empty:
                            break
                    raise RuntimeError(f"Distributed worker failure {failures}:\n"+"\n".join(tail[-40:]))
                if stopped_at and time.monotonic()-stopped_at>180:
                    raise RuntimeError("Distributed workers did not complete graceful cancellation within 180 seconds")
                try:
                    rank,line=lines.get(timeout=.2)
                except queue.Empty:
                    continue
                publish(rank,line)
            except KeyboardInterrupt:
                # Ctrl+C at the launcher becomes the same synchronized stop
                # request used by the UI. A second Ctrl+C forces termination.
                if stopped_at:
                    raise
                shared_stop.parent.mkdir(parents=True,exist_ok=True)
                shared_stop.touch();stopped_at=time.monotonic()
        failures=[(rank,p.wait()) for rank,p in enumerate(processes) if p.wait()!=0]
        if failures:
            raise RuntimeError(f"Distributed worker failure {failures}:\n"+"\n".join(tail[-40:]))
        if not result_path.is_file():
            raise RuntimeError("Distributed workers exited without a rank-0 result")
        return json.loads(result_path.read_text(encoding="utf-8"))
    except BaseException:
        for process in processes:
            _terminate_process_tree(process)
        raise
    finally:
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _terminate_process_tree(process)
            if process.stdout:
                process.stdout.close()


def launch_training(config: dict, progress=None, stop_event=None, stop_file: str | None = None) -> dict:
    """Run locally, or launch one synchronized worker per selected GPU.

    Environment selection happens before importing PyTorch in this function.
    The UI's stop event becomes a shared stop file read by all ranks. Normal
    stop saves a rank-0 resumable checkpoint; hung children are terminated only
    after the graceful shutdown deadline.
    """
    if stop_file:
        previous_event, shared_path = stop_event, Path(stop_file).expanduser().resolve()
        class CombinedStop:
            def is_set(self):
                return shared_path.exists() or (previous_event is not None and previous_event.is_set())
        stop_event = CombinedStop()
    if not config.get("multi_gpu",False) or int(os.environ.get("WORLD_SIZE","1")) > 1:
        from .training import run_training
        return run_training(config,progress=progress,stop_event=stop_event)
    from .config import validate_config
    config = validate_config(config,check_paths=False)
    if not Path(config["pairs_manifest"]).is_file():
        raise FileNotFoundError(f"Preference manifest not found: {config['pairs_manifest']}")
    gpu_ids = config.get("gpu_ids",["0","1"])
    if isinstance(gpu_ids,str):
        gpu_ids = [part.strip() for part in gpu_ids.split(",") if part.strip()]
    gpu_ids = [str(value) for value in gpu_ids]
    if len(gpu_ids) < 2 or len(set(gpu_ids)) != len(gpu_ids) or any(not value.isdigit() for value in gpu_ids):
        raise ValueError("Multi-GPU training needs at least two distinct GPU IDs, for example ['0','1']")
    if str(config.get("device","auto")).split(":")[0] not in ("auto","cuda"):
        raise ValueError("multi_gpu requires CUDA GPUs; use torchrun directly for CPU distributed tests")
    # Resolve potentially large downloads before workers wait at a collective.
    # This helper uses no CUDA context; the parent keeps GPU memory free.
    from .training import resolve_training_model
    config = resolve_training_model(config,progress)
    root = Path(config["output_dir"]).expanduser().resolve()
    root.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jk-step-distributed-",dir=root.parent) as temporary:
        folder = Path(temporary)
        config_path = folder / "worker_config.json"
        config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
        shared_stop = Path(stop_file).expanduser().resolve() if stop_file else folder / "stop.requested"
        result_path = folder / "result.json"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        env["USE_LIBUV"] = "0"
        env.setdefault("OMP_NUM_THREADS","1")
        env["PYTHONUNBUFFERED"] = "1"
        backend = config.get("distributed_backend","auto")
        local = backend=="local" or (backend=="auto" and sys.platform=="win32")
        if progress:
            progress({"event":"distributed_launch","world_size":len(gpu_ids),"gpu_ids":gpu_ids,
                      "launcher":"local_shared_memory" if local else (
                          "filestore_subprocess" if sys.platform=="win32" else "torchrun"),
                      "message":"Launching synchronous LoRA data parallel workers"})
        if local:
            with start_local_broker(len(gpu_ids)) as broker:
                env.update(broker.worker_environment())
                return _launch_windows_workers(config_path,shared_stop,result_path,folder/"unused.store",
                                               gpu_ids,env,progress,stop_event)
        if sys.platform=="win32":
            return _launch_windows_workers(config_path,shared_stop,result_path,folder/"gloo.store",
                                            gpu_ids,env,progress,stop_event)
        command = [sys.executable,"-m","torch.distributed.run","--standalone",
                   "--nnodes=1",f"--nproc_per_node={len(gpu_ids)}","-m","jk_step.distributed",
                   "--config",str(config_path),"--stop-file",str(shared_stop),"--result",str(result_path)]
        process = subprocess.Popen(command,cwd=str(Path(__file__).resolve().parent.parent),env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=="win32" else 0,
            start_new_session=sys.platform!="win32")
        lines = queue.Queue()
        def reader():
            try:
                for line in process.stdout:
                    lines.put(line.rstrip())
            finally:
                lines.put(None)
        threading.Thread(target=reader,daemon=True).start()
        tail, stopped_at, finished = [], None, False
        try:
            while not finished or process.poll() is None:
                try:
                    if (stop_event is not None and stop_event.is_set()) or shared_stop.exists():
                        shared_stop.parent.mkdir(parents=True,exist_ok=True)
                        shared_stop.touch()
                        stopped_at = stopped_at or time.monotonic()
                        if time.monotonic()-stopped_at > 180:
                            _terminate_process_tree(process)
                            raise RuntimeError("Distributed workers did not complete graceful cancellation within 180 seconds")
                    try:
                        line = lines.get(timeout=.2)
                    except queue.Empty:
                        continue
                    if line is None:
                        finished = True
                        continue
                    tail.append(line)
                    tail = tail[-80:]
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        record = {"event":"distributed_log","message":line}
                    if progress:
                        progress(record)
                except KeyboardInterrupt:
                    if stopped_at:
                        raise
                    shared_stop.parent.mkdir(parents=True,exist_ok=True)
                    shared_stop.touch();stopped_at=time.monotonic()
            code = process.wait()
            if code != 0:
                raise RuntimeError(f"Distributed training exited with code {code}:\n"+"\n".join(tail[-30:]))
            if not result_path.is_file():
                raise RuntimeError("Distributed workers exited without a rank-0 result")
            return json.loads(result_path.read_text(encoding="utf-8"))
        except BaseException:
            _terminate_process_tree(process)
            raise
        finally:
            if process.stdout:
                process.stdout.close()


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="JK-Step distributed rank worker")
    parser.add_argument("--config",required=True)
    parser.add_argument("--stop-file",required=True)
    parser.add_argument("--result",required=True)
    parser.add_argument("--local-rank",type=int,default=None)
    args = parser.parse_args(argv)
    if args.local_rank is not None:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    class SharedStop:
        def is_set(self):
            return Path(args.stop_file).exists()
    from .training import run_training
    rank = int(os.environ.get("RANK","0"))
    def progress(record):
        print(json.dumps(record,ensure_ascii=False,allow_nan=False),flush=True)
    result = run_training(config,progress=progress if rank==0 else None,stop_event=SharedStop())
    if rank==0:
        Path(args.result).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
