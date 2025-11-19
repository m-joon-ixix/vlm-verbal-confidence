import re
from typing import List


def split_to_sentences(text: str) -> List[str]:
    # delimiter: punctuation followed by one or more whitespaces
    return re.split(r"(?<=[.!?])\s+", text)
