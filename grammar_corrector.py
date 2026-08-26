"""
grammar_corrector.py

Loads the fine-tuned T5-base grammar correction model and exposes a
single function to correct a sentence's grammar.

Model path assumes you've downloaded the 'phase6b_t5base_finetuned' folder
(saved via model.save_pretrained() / tokenizer.save_pretrained()) from
Google Drive and placed it alongside this file, or you update MODEL_PATH
below to point at wherever you've stored it locally / on disk.
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# --- Config ---
MODEL_PATH = "./phase6b_t5base_finetuned"  # update this path if your model folder lives elsewhere
MAX_INPUT_LEN = 32
MAX_OUTPUT_LEN = 32

# --- Load model + tokenizer once at import time (cached across calls) ---
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
_model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(_device)
_model.eval()


def correct_grammar(text: str, num_beams: int = 5) -> str:
    """
    Takes a raw sentence (or short passage) and returns the grammar-corrected
    version using the fine-tuned T5-base model.

    Args:
        text: input sentence to correct.
        num_beams: beam search width (higher = more thorough search, slower).

    Returns:
        The corrected sentence as a string. Returns an empty string if
        input is empty/whitespace-only.
    """
    text = text.strip()
    if not text:
        return ""

    input_text = f"grammar: {text}"
    input_ids = _tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN,
    ).input_ids.to(_device)

    with torch.no_grad():
        output_ids = _model.generate(
            input_ids,
            max_length=MAX_OUTPUT_LEN,
            num_beams=num_beams,
            early_stopping=True,
        )

    corrected = _tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return corrected
