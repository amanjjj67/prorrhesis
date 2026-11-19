"""Background jobs for the refresh endpoint (spec section 14.1).

`POST /api/refresh` starts the ingest + predict flow in a worker thread and
returns a job id right away, so the browser never waits on a long download.
`GET /api/jobs/{id}` reports what the job is doing.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    kind: str
    status: str = "pending"
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    steps: list[dict] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None

    def step(self, name: str, ok: bool = True, detail: str = "") -> None:
        self.steps.append({"name": name, "ok": bool(ok), "detail": detail,
                           "at": _now()})

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "steps": list(self.steps),
            "result": self.result,
            "error": self.error,
        }


class JobManager:
    """Keeps the jobs of this process. Jobs die with the process."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, target: Callable[[Job], dict]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, created_at=_now())
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job, target), daemon=True).start()
        return job

    @staticmethod
    def _run(job: Job, target: Callable[[Job], dict]) -> None:
        job.status = "running"
        job.started_at = _now()
        try:
            job.result = target(job)
            job.status = "done"
        except Exception as exc:  # a failed job must not take the server down
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.finished_at = _now()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)


MANAGER = JobManager()
