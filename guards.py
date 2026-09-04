#guards.py
OLCHIKI_RANGE = (0x1C50, 0x1C7F)

def is_valid_olchiki(text: str) -> bool:
    if not text:
        return False
    return all(OLCHIKI_RANGE[0] <= ord(ch) <= OLCHIKI_RANGE[1] for ch in text if not ch.isspace())
