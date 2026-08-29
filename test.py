import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

def test_hindi_to_santhali():
    print("[1/3] Initializing IndicTrans2 Engine...")
    model_name = "ai4bharat/indictrans2-indic-indic-dist-320M"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    ip = IndicProcessor(inference=True)

    print("[2/3] Processing Translation Pipeline...")
    src_lang, tgt_lang = "hin_Deva", "sat_Olck"
    hindi_text = input("Enter Hindi text to translate to Santhali: ")

    # Preprocess text (handles language tags)
    batch = ip.preprocess_batch([hindi_text], src_lang=src_lang, tgt_lang=tgt_lang)
    
    # Tokenize input batch
    inputs = tokenizer(
        batch, 
        truncation=True, 
        padding="longest", 
        return_tensors="pt"
    )

    # Generate translation sequence
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=False,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1
        )

    # Decode and postprocess
    decoded_tokens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    translations = ip.postprocess_batch(decoded_tokens, lang=tgt_lang)

    print("\n" + "="*50)
    print(f"Input Hindi     : {hindi_text}")
    print(f"Santhali Output : {translations[0]}")
    print("="*50)

if __name__ == "__main__":
    test_hindi_to_santhali()