import hashlib
import time
from typing import Optional

from .config import get_settings


class ABRouter:
    def __init__(self, default_profile: str = "control") -> None:
        self.settings = get_settings()
        self.default_profile = default_profile

    def choose(self, requested_profile: Optional[str], sticky_key: Optional[str] = None) -> str:
        if requested_profile and requested_profile in self.settings.ab_profiles:
            return requested_profile

        key = sticky_key or str(time.time())
        bucket = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100

        if bucket < 10 and "quality" in self.settings.ab_profiles:
            return "quality"
        if bucket < 40 and "fast" in self.settings.ab_profiles:
            return "fast"
        return self.default_profile
