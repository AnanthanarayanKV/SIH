#api.py
import io
import os
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from translator import translator
from guards import is_valid_olchiki

app = FastAPI(title="Hindi-to-Santali Translation + TTS")

AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)


# --- TTS with graceful fallback ---------------------------------------------
# The TTS model (ai4bharat/indic-parler-tts) is a gated HF repo. If it can't be
# loaded, we degrade gracefully so the rest of the API still works.
try:
    from tts import tts
    TTS_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    tts = None
    TTS_AVAILABLE = False
    _TTS_LOAD_ERROR = str(exc)
# -----------------------------------------------------------------------------


class TranslateRequest(BaseModel):
    sentences: list[str]


class SpeakRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tts_available": TTS_AVAILABLE,
    }


@app.post("/translate")
def translate(req: TranslateRequest):
    start = time.time()
    outputs = translator.translate(req.sentences)
    latency = time.time() - start

    results = []
    for hindi, santali in zip(req.sentences, outputs):
        results.append({
            "hindi": hindi,
            "santali": santali,
            "uncertain": not is_valid_olchiki(santali),
        })

    return {"results": results, "latency_s": latency}


@app.post("/speak")
def speak(req: SpeakRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    if not TTS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="TTS is unavailable. The model 'ai4bharat/indic-parler-tts' "
                   "is gated and could not be loaded. Request access on "
                   "Hugging Face and re-run.",
        )

    filename = f"speak_{uuid.uuid4().hex[:8]}.wav"
    filepath = os.path.join(AUDIO_DIR, filename)

    tts.synthesize(req.text, output_path=filepath)

    return FileResponse(filepath, media_type="audio/wav", filename=filename)


@app.get("/")
def root():
    return {
        "message": "Hindi-to-Santali translation + TTS service.",
        "endpoints": ["GET /health", "POST /translate", "POST /speak"],
    }
