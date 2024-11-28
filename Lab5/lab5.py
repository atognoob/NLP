# Preparation
from sys import argv
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# Setting seed for repeatable results
np.random.seed(42)
torch.manual_seed(42)

# Loading
def load_tokenizer_and_model(model_name_or_path):
    return GPT2Tokenizer.from_pretrained(model_name_or_path), GPT2LMHeadModel.from_pretrained(model_name_or_path)

# Launch
tok, model = load_tokenizer_and_model("sberbank-ai/rugpt3large_based_on_gpt2")
def generate(
    model, tok, text,
    do_sample=True, max_length=100, repetition_penalty=5.0,
    top_k=5, top_p=0.95, temperature=1,
    num_beams=None,
    no_repeat_ngram_size=3
):
    input_ids = tok.encode(text, return_tensors="pt")
    out = model.generate(
        input_ids,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        top_k=top_k, top_p=top_p, temperature=temperature,
        num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size
    )
    return list(map(tok.decode, out))

generated = generate(model, tok, "Тяжелые виды труда механизируются, в домах работают лифты, а в квартирах – посудомоечные и стиральные машины, мы все больше времени проводим перед", num_beams=10)
print(generated[0])


