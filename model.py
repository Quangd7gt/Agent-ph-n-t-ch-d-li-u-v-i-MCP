import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class GemmaModel:
    def __init__(self, model_name="google/gemma-2b"):
        token = os.getenv("HUGGINGFACE_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            device_map="auto",
            torch_dtype=torch.float16
        )

    def generate_text(self, prompt: str, max_new_tokens: int = 160) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

