import re
from typing import List


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"\b\w+\b", text)]
