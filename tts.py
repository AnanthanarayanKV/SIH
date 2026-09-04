#tts.py
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

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

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

# Load once, import this instance everywhere
tts = TextToSpeech()
