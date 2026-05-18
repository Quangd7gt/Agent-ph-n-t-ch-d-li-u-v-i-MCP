from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEAD_PROXY_VALUES = {
    "http://127.0.0.1:9",
    "https://127.0.0.1:9",
    "http://localhost:9",
    "https://localhost:9",
}
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def clear_dead_proxy_env() -> None:
    for key in PROXY_ENV_KEYS:
        value = os.getenv(key)
        if value and value.rstrip("/").lower() in DEAD_PROXY_VALUES:
            os.environ.pop(key, None)


class GemmaModel:
    def __init__(self, model_name: str = "google/gemma-2b-it"):
        clear_dead_proxy_env()
        token = os.getenv("HUGGINGFACE_TOKEN") or None
        self.device = self._resolve_device(os.getenv("GEMMA_DEVICE", "auto"))
        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=token,
                torch_dtype=torch_dtype,
            )
        except OSError as exc:
            message = str(exc).lower()
            if "gated repo" in message or "401" in message or "unauthorized" in message:
                raise RuntimeError(
                    "Cannot load Gemma from Hugging Face. Make sure your "
                    "HUGGINGFACE_TOKEN has read access and that you accepted "
                    f"the model terms for {model_name}."
                ) from exc
            raise
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        if device_name and device_name.lower() != "auto":
            return torch.device(device_name)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.inference_mode()
    def generate_text(self, prompt: str, max_new_tokens: int = 160) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
