# test_pipeline.py
"""
Test script that runs the full pipeline against known cases.
Logs latency per call and prints a summary table.
"""
import sys
import time
from fastapi.testclient import TestClient
from main import app

# Force UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

client = TestClient(app)

# KNOWN GOOD: should pass guard (valid Ol Chiki output)
KNOWN_GOOD = ["पानी", "माँ", "किताब", "रात", "खाना", "बाड़ा"]

# KNOWN BAD (wrong-script): should fail guard (non-Ol-Chiki output like Arabic)
KNOWN_BAD_WRONG_SCRIPT = ["सुबह"]

# KNOWN WRONG-BUT-VALID: these produce valid Ol Chiki but wrong translations
# (known limitation - guard correctly passes them, length-ratio heuristic planned)
KNOWN_WRONG_BUT_VALID = ["नमस्कार", "दो"]

results = []

def test_translate(text: str):
    """Test /translate endpoint"""
    start = time.time()
    r = client.post("/translate", json={"text": text})
    latency_ms = int((time.time() - start) * 1000)
    return r, latency_ms

def test_speak(text: str):
    """Test /speak endpoint"""
    start = time.time()
    r = client.post("/speak", json={"text": text})
    latency_ms = int((time.time() - start) * 1000)
    return r, latency_ms

def safe_repr(text: str) -> str:
    """Return a safe representation for printing."""
    try:
        return text
    except UnicodeEncodeError:
        return text.encode('unicode_escape').decode('ascii')

def run_case(hindi: str, category: str, expect_valid: bool):
    """Run a single test case."""
    # Test /translate
    r_translate, lat_translate = test_translate(hindi)
    translate_data = r_translate.json()
    
    # Test /speak
    r_speak, lat_speak = test_speak(hindi)
    tts_attempted = r_speak.status_code == 200
    
    actual_valid = translate_data.get("valid", False)
    guard_ok = (actual_valid == expect_valid)
    
    results.append({
        "input": hindi,
        "translated": translate_data.get("santali", ""),
        "valid": actual_valid,
        "latency_ms": lat_translate + lat_speak,
        "tts_attempted": tts_attempted,
        "category": category,
        "expect_valid": expect_valid,
        "guard_ok": guard_ok
    })
    
    status = "OK" if guard_ok else "FAIL"
    tts_status = "YES" if tts_attempted else "NO"
    if category == "WRONG_SCRIPT" and tts_attempted:
        tts_status += " *** BUG ***"
    santali_safe = safe_repr(translate_data.get('santali', ''))
    print(f"  {status} {hindi:<10} -> {santali_safe:<25} valid={actual_valid} (exp={expect_valid})  latency={lat_translate+lat_speak}ms  TTS={tts_status}")
    
    return guard_ok, tts_attempted

print("=" * 110)
print("TESTING KNOWN GOOD CASES (should pass guard, should produce audio)")
print("=" * 110)

all_ok = True
for hindi in KNOWN_GOOD:
    guard_ok, tts_attempted = run_case(hindi, "GOOD", expect_valid=True)
    if not guard_ok or not tts_attempted:
        all_ok = False

print("\n" + "=" * 110)
print("TESTING KNOWN BAD - WRONG SCRIPT (should fail guard, should NOT attempt TTS)")
print("=" * 110)

for hindi in KNOWN_BAD_WRONG_SCRIPT:
    guard_ok, tts_attempted = run_case(hindi, "WRONG_SCRIPT", expect_valid=False)
    if not guard_ok or tts_attempted:
        all_ok = False

print("\n" + "=" * 110)
print("TESTING KNOWN WRONG-BUT-VALID (known limitation: guard passes, TTS runs)")
print("=" * 110)

for hindi in KNOWN_WRONG_BUT_VALID:
    guard_ok, tts_attempted = run_case(hindi, "WRONG_BUT_VALID", expect_valid=True)
    # These are expected to pass guard and run TTS (known limitation)
    if not guard_ok or not tts_attempted:
        all_ok = False

print("\n" + "=" * 110)
print("SUMMARY TABLE")
print("=" * 110)
print(f"{'Input':<12} {'Translated Output':<28} {'Guard Valid?':<12} {'Exp Valid?':<10} {'Latency (ms)':<14} {'TTS Attempted?'}")
print("-" * 110)

for r in results:
    tts_str = "YES" if r["tts_attempted"] else "NO"
    if r["category"] == "WRONG_SCRIPT" and r["tts_attempted"]:
        tts_str += " *** BUG ***"
    translated_safe = safe_repr(r['translated'])
    print(f"{r['input']:<12} {translated_safe:<28} {str(r['valid']):<12} {str(r['expect_valid']):<10} {r['latency_ms']:<14} {tts_str}")

print("-" * 110)

if all_ok:
    print("\nALL TESTS PASSED: Gating logic works correctly.")
    print("- Wrong-script output (सुबह) correctly blocked from TTS")
    print("- Valid Ol Chiki output (KNOWN GOOD + KNOWN WRONG-BUT-VALID) correctly allowed TTS")
    print("- Wrong-but-valid-Ol-Chiki (नमस्कार, दो) passes guard (known limitation, TODO: length-ratio heuristic)")
else:
    print("\nSOME TESTS FAILED: Check gating logic.")

# Assertions for critical bugs
for r in results:
    if r["category"] == "WRONG_SCRIPT" and r["tts_attempted"]:
        raise AssertionError(f"CRITICAL BUG: TTS attempted on WRONG SCRIPT input '{r['input']}'")

print("\nCritical assertions passed.")