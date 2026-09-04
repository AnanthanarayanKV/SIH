#test_tts.py
import os
import time
from translator import translator
from tts import tts
from guards import is_valid_olchiki

HINDI_INPUTS = ["पानी", "माँ", "किताब", "रात", "सुबह"]

AUDIO_DIR = "test_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

results = []

for hindi in HINDI_INPUTS:
    print(f"\n=== {hindi} ===")

    t0 = time.time()
    santali = translator.translate([hindi])[0]
    t_translate = time.time() - t0

    safe_name = hindi.encode("ascii", "ignore").decode() or f"w{len(results)}"
    out_path = os.path.join(AUDIO_DIR, f"{safe_name}_{hindi}.wav")

    t1 = time.time()
    audio_arr, sr = tts.synthesize(santali, output_path=out_path)
    t_tts = time.time() - t1
    t_total = time.time() - t0

    valid = is_valid_olchiki(santali)
    print(f"  Hindi:      {hindi}")
    print(f"  Santali:    {santali}   (valid_olchiki={valid})")
    print(f"  Translate:  {t_translate:.3f}s")
    print(f"  TTS:        {t_tts:.3f}s  (duration={len(audio_arr)/sr:.2f}s)")
    print(f"  TOTAL:      {t_total:.3f}s")
    print(f"  Audio:      {out_path}")

    results.append((hindi, santali, valid, t_total, out_path))

print("\n\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"{'Hindi input':<14} {'Santali text':<20} {'latency (s)':<12} {'audio file'}")
print("-" * 80)
for hindi, santali, valid, lat, path in results:
    match = " (bad?)" if not valid else ""
    print(f"{hindi:<14} {santali:<20} {lat:<12.3f} {path}{match}")

avg = sum(r[3] for r in results) / len(results)
print("-" * 80)
print(f"{'AVERAGE':<14} {'':<20} {avg:<12.3f}")
print(f"\nTarget end-to-end latency: < 3.0 s per item")
