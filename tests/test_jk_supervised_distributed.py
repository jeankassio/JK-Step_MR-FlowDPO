"""Two CPU workers exercise synchronized normal SFT, stop and exact resume."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from jk_step.distributed import start_local_broker


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def sft_distributed_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("sft-real-distributed")
    workers = []
    with start_local_broker(2) as broker:
        try:
            for rank in range(2):
                env = {**os.environ, **broker.worker_environment(), "CUDA_VISIBLE_DEVICES": "-1",
                       "OMP_NUM_THREADS": "1", "RANK": str(rank), "WORLD_SIZE": "2", "LOCAL_RANK": str(rank)}
                options = {"cwd": str(ROOT), "env": env, "stdout": subprocess.PIPE,
                           "stderr": subprocess.STDOUT, "text": True, "encoding": "utf-8", "errors": "replace"}
                if os.name == "nt":
                    options["creationflags"] = subprocess.CREATE_NO_WINDOW
                workers.append(subprocess.Popen([sys.executable, str(ROOT / "tests" / "sft_fixture_runner.py"),
                                                  "--directory", str(directory)], **options))
            outputs = [process.communicate(timeout=100)[0] for process in workers]
            assert all(process.returncode == 0 for process in workers), "\n".join(outputs)
            yield [json.loads((directory / f"sft-rank-{rank}.json").read_text()) for rank in range(2)]
        finally:
            for process in workers:
                if process.poll() is None:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                       creationflags=subprocess.CREATE_NO_WINDOW,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                    else:
                        process.terminate()
                    process.wait(timeout=10)


def test_sft_gradient_reduction_and_replicas(sft_distributed_run):
    for report in sft_distributed_run:
        assert report["gradient_error"] < 2e-7
        for trace in report["engine"]["traces"].values():
            assert len(set(trace["replica_hashes"])) == 1


def test_sft_stop_checkpoint_and_exact_resume(sft_distributed_run):
    for report in sft_distributed_run:
        engine = report["engine"]
        assert engine["results"]["stopped"]["status"] == "stopped"
        assert engine["results"]["stopped"]["step"] == 2
        assert engine["results"]["resumed"]["step"] == 3
        assert engine["objective"] == "sft" and engine["rank_rng_count"] == 2
        assert engine["whole_hash"] == engine["resume_hash"]
    assert all(sft_distributed_run[0]["engine"]["saves"].values())
    assert not any(sft_distributed_run[1]["engine"]["saves"].values())


def test_sft_validation_uses_global_sample_mean(sft_distributed_run):
    validation = [row for row in sft_distributed_run[0]["engine"]["traces"]["whole"]["progress"]
                  if row["event"] == "validation"][-1]
    total, count = 0, 0
    ids = []
    for rank, report in enumerate(sft_distributed_run):
        rows = report["engine"]["traces"]["whole"]["validation"][-(2 if rank == 0 else 1):]
        for row in rows:
            ids.extend(row["ids"])
            count += row["count"]
            total += row["loss"] * row["count"]
    assert set(ids) == {"8", "9", "10"} and count == 3
    assert validation["validation_loss"] == pytest.approx(total / count, rel=1e-6)
    assert validation["objective"] == "sft" and "validation_dpo" not in validation
