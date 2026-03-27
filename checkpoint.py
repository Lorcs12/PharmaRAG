import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from logger import get_logger

log = get_logger("checkpoint")


class Checkpoint:
    def __init__(self, drug_set: str, checkpoint_dir: str = "./checkpoints"):
        self.drug_set = drug_set.lower()
        self.path     = Path(checkpoint_dir) / f"{self.drug_set}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state   = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                state = json.loads(self.path.read_text())
                log.info(
                    f"Resuming '{self.drug_set}' — "
                    f"{len(state.get('completed_set_ids', []))} labels done, "
                    f"{state.get('total_indexed', 0)} docs indexed",
                    extra={"drug_set": self.drug_set}
                )
                return state
            except Exception as e:
                log.warning(f"Corrupt checkpoint for '{self.drug_set}', starting fresh: {e}")
        return {
            "drug_set":           self.drug_set,
            "completed_set_ids":  [],
            "completed_phases":   [],
            "total_indexed":      0,
            "last_updated":       None,
        }

    def _save(self):
        self._state["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.path.write_text(json.dumps(self._state, indent=2))

    # ── Public API ─────────────────────────────────────────────────────────────

    def is_set_id_done(self, set_id: str) -> bool:
        return set_id in self._state["completed_set_ids"]

    def mark_set_id_done(self, set_id: str, drug_name: str, n_docs: int):
        if set_id not in self._state["completed_set_ids"]:
            self._state["completed_set_ids"].append(set_id)
        self._state["total_indexed"] += n_docs
        self._save()
        log.info(
            f"Label done: {drug_name} ({set_id}) +{n_docs} docs",
            extra={"drug_set": self.drug_set, "set_id": set_id,
                   "drug_name": drug_name, "n_docs": n_docs}
        )

    def is_phase_done(self, phase: str) -> bool:
        return phase in self._state["completed_phases"]

    def mark_phase_done(self, phase: str):
        if phase not in self._state["completed_phases"]:
            self._state["completed_phases"].append(phase)
        self._save()
        log.info(
            f"Phase done: {phase}",
            extra={"drug_set": self.drug_set, "phase": phase}
        )

    @property
    def total_indexed(self) -> int:
        return self._state["total_indexed"]

    def reset(self):
        """Delete checkpoint — forces full re-ingestion."""
        if self.path.exists():
            self.path.unlink()
        self._state = self._load()
        log.info(f"Checkpoint reset for '{self.drug_set}'",
                 extra={"drug_set": self.drug_set})