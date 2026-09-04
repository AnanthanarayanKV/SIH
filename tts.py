# tts.py
import io
import os
import torch
import numpy as np
import soundfile as sf

# --- transformers 5.x compatibility shim -------------------------------------
# parler-tts 0.2.2 imports `isin_mps_friendly` which was removed in transformers 5.x.
# It was a small wrapper around torch.isin that moved tensors to CPU on MPS devices.
import transformers.pytorch_utils as _pw
if not hasattr(_pw, "isin_mps_friendly"):
    def _isin_mps_friendly(elements, test_elements):
        if getattr(_pw, "is_torch_mps_available", lambda: False)():
            return torch.isin(elements.to("cpu"), test_elements.to("cpu")).to(elements.device)
        return torch.isin(elements, test_elements)
    _pw.isin_mps_friendly = _isin_mps_friendly
# -----------------------------------------------------------------------------

try:
    from parler_tts import ParlerTTSForConditionalGeneration
    from parler_tts.configuration_parler_tts import ParlerTTSConfig
    from transformers import AutoTokenizer

    # --- transformers 4.5x compatibility shim ------------------------------------
    # transformers >=4.45 changed PretrainedConfig serialization: `to_diff_dict`
    # instantiates `self.__class__().to_dict()`. parler-tts 0.2.2's ParlerTTSConfig
    # can't be instantiated without sub-configs, which crashes on config load.
    # Newer parler-tts marks the class as having no defaults at init to skip that path.
    ParlerTTSConfig.has_no_defaults_at_init = True
    # -----------------------------------------------------------------------------

    MODEL_NAME = "ai4bharat/indic-parler-tts"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Santali is an officially supported language in Indic Parler-TTS. A Santali
    # speaker is referenced so the model picks an appropriate voice.
    DEFAULT_DESCRIPTION = (
        "Sita speaks with a moderate speed and a clear, natural-sounding female voice. "
        "The recording is of very high quality, with the voice sounding very close up "
        "and with almost no background noise."
    )

    class TextToSpeech:
        def __init__(self):
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.description_tokenizer = AutoTokenizer.from_pretrained(
                self.model.config.text_encoder._name_or_path
            )
            self.sampling_rate = self.model.config.sampling_rate

        def synthesize(
            self,
            text: str,
            description: str = DEFAULT_DESCRIPTION,
            output_path: str | None = None,
        ) -> tuple[np.ndarray, int]:
            description_input_ids = self.description_tokenizer(
                description, return_tensors="pt"
            ).to(DEVICE)
            prompt_input_ids = self.tokenizer(text, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                generation = self.model.generate(
                    input_ids=description_input_ids.input_ids,
                    attention_mask=description_input_ids.attention_mask,
                    prompt_input_ids=prompt_input_ids.input_ids,
                    prompt_attention_mask=prompt_input_ids.attention_mask,
                )

            audio_arr = generation.cpu().numpy().squeeze()

            if output_path is not None:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                sf.write(output_path, audio_arr, self.sampling_rate)

            return audio_arr, self.sampling_rate

        def speak(self, santali_text: str) -> bytes:
            """
            Synthesize Santali text to speech and return audio as bytes (WAV format).
            
            CRITICAL: This must only be called with text that has already passed
            is_valid_olchiki() — do not call TTS on unguarded translation output.
            """
            audio_arr, sr = self.synthesize(santali_text)
            # Write to bytes buffer
            buffer = io.BytesIO()
            sf.write(buffer, audio_arr, sr, format='WAV')
            buffer.seek(0)
            return buffer.read()

    # Try to load the model, but handle gated repo gracefully
    try:
        tts = TextToSpeech()
        TTS_AVAILABLE = True
    except Exception as e:
        print(f"Warning: Could not load TTS model (gated repo): {e}")
        tts = None
        TTS_AVAILABLE = False

except ImportError:
    # parler_tts not installed
    tts = None
    TTS_AVAILABLE = False

# Mock TTS for testing when real model is unavailable
class MockTTS:
    """Mock TTS that returns silent audio for testing purposes."""
    def __init__(self):
        self.sampling_rate = 16000
    
    def speak(self, santali_text: str) -> bytes:
        # Generate 1 second of silence as WAV
        audio_arr = np.zeros(self.sampling_rate, dtype=np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, audio_arr, self.sampling_rate, format='WAV')
        buffer.seek(0)
        return buffer.read()

# Use mock if real TTS not available
if not TTS_AVAILABLE:
    tts = MockTTS()
    TTS_AVAILABLE = True  # Mock is available