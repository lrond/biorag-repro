from __future__ import annotations

import re
import time
from pathlib import Path

from biorag.config import ProjectConfig, resolve_model_source
from biorag.io import dump_jsonl
from biorag.prompting import build_prompt
from biorag.types import PredictionRecord, QuestionRecord, RetrievedContext
from biorag.utils import get_logger, normalize_text

LOGGER = get_logger(__name__)


def _rule_based_generate(question: QuestionRecord, context: RetrievedContext) -> str:
    combined = "\n".join(candidate.text for candidate in context.candidates)
    lower = combined.lower()
    if not combined.strip():
        return "I don't know."
    if question.type == "yesno":
        if " yes " in f" {lower} " or "positive" in lower:
            return "Yes"
        if " no " in f" {lower} " or "negative" in lower:
            return "No"
        return "I don't know."
    explicit = re.search(r"(?:answer|exact answer)\s*[:\-]\s*([^\n.;]+)", combined, re.IGNORECASE)
    if explicit:
        return explicit.group(1).strip()
    if question.type == "list":
        bullet_like = re.findall(r"(?:^|\n)[\-\*\d\.\)]\s*([^\n]+)", combined)
        if bullet_like:
            return "\n".join(item.strip() for item in bullet_like[:5])
        sentences = [sentence.strip() for sentence in re.split(r"[.;]", combined) if sentence.strip()]
        return "\n".join(sentences[:3])
    if question.type == "summary":
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", combined) if sentence.strip()]
        return " ".join(sentences[:2]) if sentences else "I don't know."
    sentences = [sentence.strip() for sentence in re.split(r"[.;]", combined) if sentence.strip()]
    return sentences[0] if sentences else "I don't know."


def _huggingface_generate(prompt: str, config: ProjectConfig, device: str) -> str:
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ModuleNotFoundError as error:
        raise RuntimeError("transformers and torch are required for generator backend.") from error
    model_source = resolve_model_source(
        config.models.generator.model_name,
        checkpoint_path=config.models.generator.checkpoint_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForCausalLM.from_pretrained(model_source)
    model.to(device)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=config.models.generator.max_new_tokens,
            do_sample=config.models.generator.do_sample,
            temperature=config.models.generator.temperature,
        )
    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    answer = full_text[len(prompt) :].strip() if full_text.startswith(prompt) else full_text.strip()
    return answer


def _postprocess_prediction(question: QuestionRecord, raw_answer: str) -> tuple[str, list[str], bool]:
    answer = raw_answer.strip() or "I don't know."
    normalized = normalize_text(answer)
    abstained = "i dont know" in normalized or normalized == "unknown"
    if question.type == "yesno":
        if normalized.startswith("yes"):
            return "Yes", [], False
        if normalized.startswith("no"):
            return "No", [], False
        return "I don't know.", [], True
    if question.type in {"factoid", "list"}:
        parts = [
            part.strip(" -\t")
            for part in re.split(r"[\n,;]", answer)
            if part.strip(" -\t")
        ]
        primary = parts[0] if parts else "I don't know."
        return primary, parts, abstained
    return answer, [], abstained


def generate_predictions(
    questions: list[QuestionRecord],
    contexts: list[RetrievedContext],
    config: ProjectConfig,
    output_path: str | Path,
    device: str,
) -> Path:
    questions_by_id = {question.id: question for question in questions}
    predictions: list[PredictionRecord] = []
    for context in contexts:
        question = questions_by_id[context.question_id]
        prompt = build_prompt(question, context, config)
        start = time.perf_counter()
        if config.inference.abstain_when_empty and not context.candidates:
            raw_answer = "I don't know."
        elif config.models.generator.backend == "rule_based":
            raw_answer = _rule_based_generate(question, context)
        else:
            raw_answer = _huggingface_generate(prompt, config, device)
        latency = time.perf_counter() - start
        answer, ranked_answers, abstained = _postprocess_prediction(question, raw_answer)
        predictions.append(
            PredictionRecord(
                question_id=question.id,
                question_type=question.type,
                answer=answer,
                ranked_answers=ranked_answers,
                retrieved_documents=[candidate.document_id for candidate in context.candidates],
                latency_seconds=latency,
                abstained=abstained,
                prompt_excerpt=prompt[:500],
            )
        )
    return dump_jsonl(output_path, predictions)
