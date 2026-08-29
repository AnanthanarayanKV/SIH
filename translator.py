#translator.py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

MODEL_NAME = "ai4bharat/indictrans2-indic-indic-dist-320M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Translator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
        self.model.eval()
        self.ip = IndicProcessor(inference=True)

    def translate(self, sentences: list[str], src_lang="hin_Deva", tgt_lang="sat_Olck") -> list[str]:
        batch = self.ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = self.tokenizer(batch, truncation=True, padding="longest", return_tensors="pt", max_length=256).to(DEVICE)

        with torch.no_grad():
            generated_tokens = self.model.generate(
            **inputs,
            use_cache=False,
            min_length=0,
            max_length=256,
            num_beams=5,
            repetition_penalty=1.3,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )

        decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return self.ip.postprocess_batch(decoded, lang=tgt_lang)

# Load once, import this instance everywhere
translator = Translator()