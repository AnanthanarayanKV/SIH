# main.py
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from translation import translate
from guards import check_translation
from tts import tts, TTS_AVAILABLE

app = FastAPI(title="Hindi-to-Santali Translation + TTS")


class TranslateRequest(BaseModel):
    text: str


class SpeakRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "tts_available": TTS_AVAILABLE}


@app.post("/translate")
def translate_endpoint(req: TranslateRequest):
    start = time.time()
    translated = translate(req.text)
    result = check_translation(req.text, translated)
    latency_ms = int((time.time() - start) * 1000)
    
    return {
        "hindi": req.text,
        "santali": result["text"],
        "valid": result["valid"],
        "warning": result["warning"],
        "latency_ms": latency_ms
    }


@app.post("/speak")
def speak_endpoint(req: SpeakRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")
    
    # Run full pipeline: translate -> check_translation -> if valid, run TTS
    translated = translate(req.text)
    result = check_translation(req.text, translated)
    
    if not result["valid"]:
        # GATE: Do NOT call TTS on invalid translation
        raise HTTPException(
            status_code=422,
            detail=result["warning"] or "Translation uncertain — please verify manually"
        )
    
    # Only call TTS if translation passed the guard
    audio_bytes = tts.speak(result["text"])
    
    return Response(content=audio_bytes, media_type="audio/wav")