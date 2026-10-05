"""Two-process CPU/IPC tests of real PEFT gradients and preference training."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from jk_step.distributed import start_local_broker


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def distributed_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("jk-real-distributed")
    # Real shared-memory collectives work even on Windows wheels whose Gloo
    # capability flag is true but no network transport can be instantiated.
    broker = start_local_broker(2)
    workers = []
    try:
        for rank in range(2):
            env = {**os.environ, **broker.worker_environment(), "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1",
                   "RANK": str(rank), "WORLD_SIZE": "2", "LOCAL_RANK": str(rank),
                   "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
            options = {"cwd": str(ROOT), "env": env, "stdout": subprocess.PIPE,
                       "stderr": subprocess.STDOUT, "text": True, "encoding": "utf-8", "errors": "replace"}
            if os.name == "nt":
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            workers.append(subprocess.Popen([sys.executable, str(ROOT / "tests" / "ddp_fixture_runner.py"),
                                             "--directory", str(directory)], **options))
        outputs = []
        for process in workers:
            output, _ = process.communicate(timeout=90)
            outputs.append((process.returncode, output))
        assert all(code == 0 for code, _ in outputs), "\n".join(f"worker exit={code}\n{output}" for code, output in outputs)
        yield [json.loads((directory / f"rank-{rank}.json").read_text("utf-8")) for rank in range(2)]
    finally:
        # Only our own fixture process trees may be terminated after failure.
        for process in workers:
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   creationflags=subprocess.CREATE_NO_WINDOW,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                else:
                    process.terminate()
                process.wait(timeout=10)
        broker.close()


def test_real_flow_dpo_gradient_reduction_matches_effective_batch(distributed_run):
    for rank in distributed_run:
        assert rank["world_size"] == 2
        assert rank["backend"] == "local"
        for mode in ("weighted_sum", "average"):
            report = rank["gradients"][mode]
            assert report["maximum_gradient_error"] < 2e-7
            assert report["gradient_norm"] > 0
            assert len(set(report["adapter_hashes"])) == 1


def test_training_replicas_remain_identical_after_updates(distributed_run):
    for rank in distributed_run:
        for scenario in ("whole", "stopped", "resumed"):
            hashes = rank["engine"]["traces"][scenario]["replica_hashes"]
            assert len(hashes) == 2 and hashes[0] == hashes[1]


def test_data_shards_cover_training_and_uneven_validation_without_leakage(distributed_run):
    training = [rank["engine"]["traces"]["whole"]["training_ids"] for rank in distributed_run]
    assert len(training[0]) == len(training[1]) == 5
    assert set(training[0]).isdisjoint(training[1])
    assert set(training[0] + training[1]) == {str(index) for index in range(10)}
    validation = []
    for rank in distributed_run:
        rows = rank["engine"]["traces"]["whole"]["validation"]
        # The first validation pass is an exact shard; unlike DistributedSampler,
        # no extra sample is repeated merely to give each rank equal length.
        first = []
        for row in rows:
            if any(value in first for value in row["ids"]):
                break
            first.extend(row["ids"])
        validation.append(first)
    assert sorted(map(len, validation)) == [2, 3]
    assert set(validation[0]).isdisjoint(validation[1])
    assert set(validation[0] + validation[1]) == {str(index) for index in range(10, 15)}
    assert set(training[0] + training[1]).isdisjoint(validation[0] + validation[1])


def test_validation_metrics_are_reduced_over_all_examples(distributed_run):
    main = distributed_run[0]["engine"]["traces"]["whole"]
    published = [record for record in main["progress"] if record["event"] == "validation"][-1]
    totals = {"validation_loss": 0.0, "validation_dpo": 0.0,
              "validation_preference_accuracy": 0.0, "validation_chosen_fm": 0.0}
    count = 0
    objective_mass = 0
    for rank in distributed_run:
        rows = rank["engine"]["traces"]["whole"]["validation"]
        local_size = 3 if rank["rank"] == 0 else 2
        for row in rows[-local_size:]:
            count += row["count"]
            objective_mass += row["weight_mass"]
            for published_name, name in (("validation_loss", "loss"), ("validation_dpo", "dpo"),
                                         ("validation_preference_accuracy", "accuracy"), ("validation_chosen_fm", "chosen_fm")):
                mass = row["weight_mass"] if name in ("loss", "dpo") else row["count"]
                totals[published_name] += row[name] * mass
    assert count == 5
    for name, value in totals.items():
        denominator = objective_mass if name in ("validation_loss", "validation_dpo") else count
        assert published[name] == pytest.approx(value / denominator, rel=1e-6, abs=1e-8)


def test_rank_one_stop_is_global_and_checkpoint_files_have_one_writer(distributed_run):
    for rank in distributed_run:
        stopped = rank["engine"]["results"]["stopped"]
        assert stopped["status"] == "stopped" and stopped["step"] == 2
        assert stopped["world_size"] == 2
    assert all(distributed_run[0]["engine"]["save_calls"].values())
    assert not any(distributed_run[1]["engine"]["save_calls"].values())


def test_distributed_resume_restores_every_rank_rng_exactly(distributed_run):
    for rank in distributed_run:
        engine = rank["engine"]
        assert engine["rank_rng_count"] == 2 and engine["rank_rng_different"]
        assert engine["checkpoint_world_size"] == 2
        assert engine["results"]["resumed"]["step"] == 3
        assert engine["whole_adapter"] == engine["resumed_adapter"]
