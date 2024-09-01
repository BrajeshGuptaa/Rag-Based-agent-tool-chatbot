import json
import os
import time
import uuid
from typing import Any, Dict

from .config import get_settings


def log_event(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    settings = get_settings()
    os.makedirs(settings.log_dir, exist_ok=True)
    entry = {
        "id": str(uuid.uuid4()),
        "type": event_type,
        "ts": time.time(),
        **payload,
    }
    logfile = os.path.join(settings.log_dir, f"{event_type}.log")
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry
