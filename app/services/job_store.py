# app/services/job_store.py
import json
import threading
from pathlib import Path

_lock = threading.Lock()
_jobs: dict = {}


def _write(job_dir: Path, data: dict):
    (job_dir / "status.json").write_text(json.dumps(data))


def create_job(job_id: str, job_dir: Path, total: int):
    data = {"status": "running", "done": 0, "total": total}
    with _lock:
        _jobs[job_id] = dict(data)
    job_dir.mkdir(parents=True, exist_ok=True)
    _write(job_dir, data)


def mark_sample_done(job_id: str, job_dir: Path, done: int):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["done"] = done
    status_file = job_dir / "status.json"
    data = json.loads(status_file.read_text())
    data["done"] = done
    _write(job_dir, data)


def complete_job(job_id: str, job_dir: Path):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "done"
    status_file = job_dir / "status.json"
    data = json.loads(status_file.read_text())
    data["status"] = "done"
    _write(job_dir, data)


def fail_job(job_id: str, job_dir: Path, error: str):
    data = {"status": "error", "message": error}
    with _lock:
        _jobs[job_id] = dict(data)
    _write(job_dir, data)


def get_job(job_id: str, job_dir: Path):
    with _lock:
        if job_id in _jobs:
            return dict(_jobs[job_id])
    status_file = job_dir / "status.json"
    if status_file.exists():
        return json.loads(status_file.read_text())
    return None
