# ground_truth_check.py
from translator import translator

def is_valid_olchiki(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if ch.isspace() or ch in "।.":
            continue
        if not (0x1C50 <= cp <= 0x1C7F):
            return False
    return True

test_pairs = [
    ("सुबह", None),
    ("रात", None),
    ("खाना", None),
    ("अलविदा", None),
    ("स्कूल", None),
    ("पढ़ाई", None),
    ("विद्यालय", None),
    ("किताब", None),
    ("बाड़ा", None),
]

for hindi, expected in test_pairs:
    output = translator.translate([hindi])[0]          # index [0] — unwrap the list
    if not is_valid_olchiki(output):
        output = "[Translation uncertain — please verify manually]"

    print(f"Hindi: {hindi} | Santali (Ol Chiki): {output}")   # inside the loop