# translation.py
import torch
import unicodedata
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

MODEL_NAME = "ai4bharat/indictrans2-indic-indic-dist-320M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer once at module import
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
_model.eval()
_ip = IndicProcessor(inference=True)


def translate(hindi_text: str) -> str:
    """
    Translate Hindi text to Santali (Ol Chiki script).
    
    Fixes applied (validated in manual testing):
    1. NFC normalization before preprocessing — fixes leaked nukta diacritic 
       codepoint in output for words like बाड़ा.
    2. use_cache=False in generate() — required due to incompatibility between
       IndicTrans2's custom trust_remote_code model and transformers Cache API
       (raises AttributeError: 'NoneType' object has no attribute 'shape').
    3. repetition_penalty=1.3 and no_repeat_ngram_size=2 — fixes repetition-loop
       degeneration failure mode seen without them.
    """
    # Fix 1: NFC normalization — fixes leaked nukta diacritic for words like बाड़ा
    hindi_text = unicodedata.normalize("NFC", hindi_text)
    
    batch = _ip.preprocess_batch([hindi_text], src_lang="hin_Deva", tgt_lang="sat_Olck")
    inputs = _tokenizer(batch, truncation=True, padding="longest", return_tensors="pt", max_length=256).to(DEVICE)

    with torch.no_grad():
        generated_tokens = _model.generate(
            **inputs,
            use_cache=False,  # Fix 2: required — True breaks due to IndicTrans2/transformers Cache API incompatibility
            min_length=0,
            max_length=256,
            num_beams=5,
            repetition_penalty=1.3,  # Fix 3: prevents repetition-loop degeneration
            no_repeat_ngram_size=2,  # Fix 3: prevents repetition-loop degeneration
            early_stopping=True,
        )

    decoded = _tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return _ip.postprocess_batch(decoded, lang="sat_Olck")[0]