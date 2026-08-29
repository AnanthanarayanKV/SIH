# ground_truth_check.py
from translator import translator

# (Hindi input, expected Santali gloss/rough check, source)
test_pairs = [
    ("नमस्कार", "Johar (ᱡᱚᱦᱟᱨ)"),
    ("पानी", None),      # water
    ("एक", None),         # one
    ("दो", None),         # two
    ("बच्चा", None),      # child
    ("विद्यालय", None),   # school
    ("माँ", None),        # mother
    ("धन्यवाद", None),    # thank you
]

for hindi, expected in test_pairs:
    output = translator.translate([hindi])[0]
    print(f"{hindi:12} -> {output}   (expected: {expected})")