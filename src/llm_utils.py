from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import GENERATOR_MODEL, MAX_NEW_TOKENS


def load_generator(device: str = "cuda") -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if device.startswith("cuda") and torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL, **kwargs)
    if "device_map" not in kwargs:
        model.to(device)
    model.eval()
    return tokenizer, model


def build_prompt(question: str, docs: list[dict[str, object]]) -> str:
    context = "\n\n".join(
        f"[Doc {index + 1}] PMID:{doc.get('pmid', '')} {doc.get('text', '')}"
        for index, doc in enumerate(docs)
    )
    return f"""You are a biomedical expert. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(
    question: str,
    docs: list[dict[str, object]],
    tokenizer: Any,
    model: Any,
    device: str = "cuda",
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    prompt = build_prompt(question, docs)
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered = prompt
    inputs = tokenizer([rendered], return_tensors="pt", truncation=True)
    if not hasattr(model, "hf_device_map"):
        inputs = {key: value.to(device) for key, value in inputs.items()}
    else:
        first_device = next(model.parameters()).device
        inputs = {key: value.to(first_device) for key, value in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    return answer or "I don't know"
