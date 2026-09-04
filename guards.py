# guards.py
OLCHIKI_RANGE = (0x1C50, 0x1C7F)

def is_valid_olchiki(text: str) -> bool:
    """Returns True if all non-whitespace characters are in the Ol Chiki Unicode block (U+1C50–U+1C7F)."""
    if not text:
        return False
    return all(OLCHIKI_RANGE[0] <= ord(ch) <= OLCHIKI_RANGE[1] for ch in text if not ch.isspace())


def check_translation(hindi_text: str, translated_text: str) -> dict:
    """
    Validate translation output and return structured result.
    
    Returns: {"text": translated_text, "valid": bool, "warning": str | None}
    
    NOTE: This only catches wrong-script output (e.g. सुबह -> Arabic script),
    not wrong-but-valid-Ol-Chiki output (e.g. नमस्कार, दो).
    # TODO: Add length-ratio heuristic as a future improvement to catch
    # wrong-but-valid-Ol-Chiki translations.
    """
    valid = is_valid_olchiki(translated_text)
    warning = None if valid else "Translation uncertain — please verify manually"
    return {
        "text": translated_text,
        "valid": valid,
        "warning": warning
    }