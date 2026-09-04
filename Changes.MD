# Changes & Fixes

This document records every bug encountered while getting the TTS pipeline
(Hindi → Santali translation → speech) working, along with the fix applied for
each.

---

## 1. `ModuleNotFoundError: No module named 'torch'` (and other packages)

**Symptom**

Running `test_tts.py` directly failed immediately:

```
ModuleNotFoundError: No module named 'torch'
```

The system Python (3.14) had no project dependencies installed.

**Root cause**

The project's dependencies live only inside a virtual environment (`venv`).
Running the test with the bare `python` command bypassed that environment.

**Fix**

Run all project code through the virtual environment's interpreter:

```bash
venv/bin/python test_tts.py
```

---

## 2. `GatedRepoError: 403 ... access to model ai4bharat/indic-parler-tts is restricted`

**Symptom**

Even with the venv active, the model failed to load:

```
huggingface_hub.errors.GatedRepoError: 403 Client Error ...
Cannot access gated repo for url
https://huggingface.co/ai4bharat/indic-parler-tts/resolve/main/config.json.
Access to model ai4bharat/indic-parler-tts is restricted and you are not in the
authorized list.
```

**Root cause**

`ai4bharat/indic-parler-tts` is a **gated (restricted-access) Hugging Face
repository**. The machine was not authenticated, and the account in use was not
on the model's authorized list.

**Fix**

1. Request access to the model on the Hub:
   https://huggingface.co/ai4bharat/indic-parler-tts
2. Authenticate the CLI with the Hugging Face token:
   ```bash
   huggingface-cli login   # or: hf auth login
   ```
3. Verified the login with `huggingface-cli whoami` (returned the account name).

---

## 3. `ValueError: Config has to be initialized with text_encoder, audio_encoder and decoder config`

**Symptom**

After login succeeded (the gated repo was now accessible and the model config
downloaded), loading the model crashed:

```
File ".../configuration_utils.py", line 839, in from_dict
    logger.info(f"Model config {config}")
File ".../parler_tts/configuration_parler_tts.py", line 243, in __init__
    raise ValueError("Config has to be initialized with text_encoder, audio_encoder and decoder config")
```

**Root cause**

The installed `parler_tts==0.2.2` sets `has_no_defaults_at_init = False` on its
`ParlerTTSConfig`. In newer transformers (`>=4.45`), `PretrainedConfig.to_diff_dict()`
calls `self.__class__().to_dict()` to compute a diff — instantiating an *empty*
`ParlerTTSConfig()`. Because an empty instance has no `text_encoder`,
`audio_encoder` and `decoder` sub-configs, that call raises mid-serialization.

**Fix**

Set `has_no_defaults_at_init = True` on `ParlerTTSConfig` so that
`to_diff_dict()` skips instantiating the empty config. Applied in `tts.py`:

```python
from parler_tts.configuration_parler_tts import ParlerTTSConfig

ParlerTTSConfig.has_no_defaults_at_init = True
```

---

## 4. `AttributeError: 'NoneType' object has no attribute 'update'` (generation_config)

**Symptom**

With the config fix in place, `generate(...)` now failed:

```
File ".../parler_tts/modeling_parler_tts.py", line 3399, in generate
    model_kwargs = generation_config.update(**kwargs)
AttributeError: 'NoneType' object has no attribute 'update'
```

**Root cause**

`model.generation_config` was `None`. In the newer transformers, generation
state is no longer attached to the model by `PreTrainedModel` the way
`parler_tts 0.2.2` expects, so `self.generation_config` resolved to `None`.

**Partial workaround tested (not durable)**

Explicitly loading and attaching the generation config worked past this step:

```python
from transformers import GenerationConfig
model.generation_config = GenerationConfig.from_pretrained("ai4bharat/indic-parler-tts")
```

This then exposed bug #5 below, proving `parler_tts 0.2.2` cannot be married to
transformers 4.57 in general — which is why the durable fix was to move to the
supported version pairing (see #6).

---

## 5. `AttributeError: ... object has no attribute '_validate_model_kwargs'`

**Symptom**

After working around `generation_config`, generation failed again:

```
AttributeError: 'ParlerTTSForConditionalGeneration' object has no attribute '_validate_model_kwargs'
```

**Root cause**

`ParlerTTSForConditionalGeneration` in `parler_tts 0.2.2` inherits directly from
`PreTrainedModel`, **not** `GenerationMixin`. Starting with transformers v4.50,
`PreTrainedModel` no longer inherits from `GenerationMixin`, so the model lost
all the generation helper methods (`_validate_model_kwargs`, etc.) that the old
`parler_tts` `generate()` calls. This is a deep, general incompatibility — not a
one-line bug.

**Fix**

Patching `parler_tts 0.2.2` against transformers 4.57 would be brittle
(unknown number of missing APIs). The correct fix is to use the officially
supported combination: latest `parler-tts` (from git) with
`transformers==4.46.1`. See #6.

---

## 6. Core dependency incompatibility: `parler_tts==0.2.2` + `transformers==4.57.6` on Python 3.14

**Symptom**

Multiple cascading failures (bugs #3, #4, #5) all trace back to one root cause:
the installed packages don't match the model's supported versions.

**Root cause**

- The `indic-parler-tts` model card specifies installing **latest parle-tts
  from git**, which pins `transformers>=4.46.1,<=4.46.1` (i.e. exactly
  `4.46.1`).
- The old environment instead had `parler_tts==0.2.2` with
  `transformers==4.57.6` — an unsupported pairing.
- The officially supported combination needs `tokenizers==0.20.3`, which has **no
  wheel for Python 3.14**. Building it from source requires the Rust toolchain
  (`cargo`), which is not installed on this machine.

**Fix**

1. Install Python 3.12 via `uv` (has wheels for every needed package):
   ```bash
   uv python install 3.12
   ```
2. Create a fresh virtual environment from Python 3.12:
   ```bash
   uv venv --python 3.12 venv312
   ```
3. Install the supported stack into it:
   ```bash
   uv pip install --python venv312/bin/python \
       "git+https://github.com/huggingface/parler-tts.git"
   ```
   This installs `parler_tts` (latest git), `transformers==4.46.1`, and
   `tokenizers==0.20.3` wheels.
4. Install the translation dependency:
   ```bash
   uv pip install --python venv312/bin/python "indictranstoolkit==1.1.1"
   ```

**Outcome**

The full pipeline now runs with `venv312/bin/python test_tts.py` and produces
valid speech audio.

> **Important:** All project code must now be run with **`venv312`**, not the old
> `venv` (which still contains the broken `parler_tts 0.2.2` +
> `transformers 4.57.6` combination on Python 3.14).

---

## 7. TTS only runs on CPU — very slow (27–51s per item)

**Symptom**

Latency is 27–51 s per item, far above the 3.0 s target.

**Root cause**

Both environments report `torch.cuda.is_available() == False`. The torch
builds are CUDA-enabled (`+cu130`) but no GPU/driver is detected at runtime, so
inference falls back to CPU.

**Status / fix**

**Not yet fixed** — requires a CUDA-capable GPU with working drivers, or
further inference optimization (e.g. streaming, SDPA, `torch.compile`) which is
beyond a dependency fix.

---

## 8. "सुबह" mistranslated to Arabic/Urdu (`صبح`) instead of Santali

**Symptom**

In the test, `सुबह` (morning) produced `صبح ᱨᱮ` — the Urdu/Arabic word
`صبح` leaked in rather than a Santali (Ol Chiki) equivalent. The guard
`is_valid_olchiki()` flagged it as invalid.

**Root cause**

Translation quality issue in the 
`ai4bharat/indictrans2-indic-indic-dist-320M` model / the translation pipeline
(`translator.py`) for this particular word — not a TTS problem.

**Status / fix**

**Not yet fixed** — this is a translation accuracy issue. Possible next steps:
choose a different input, verify the source/target language codes
(`hin_Deva` → `sat_Olck`), or fine-tune/adjust the decoding parameters in
`translator.py`.

---

## Summary of files changed

| File | Change |
|------|--------|
| `tts.py` | Added `ParlerTTSConfig.has_no_defaults_at_init = True` compat shim (#3) |
| new `venv312/` | Fresh Python 3.12 environment with supported package versions (#6) |
| new `Changes.MD` | This document |

No other source files were changed.
