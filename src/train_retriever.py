from __future__ import annotations

import argparse
import json
import random

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.util import batch_to_device
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BASE_RETRIEVER_MODEL,
    FINETUNED_RETRIEVER_DIR,
    LEARNING_RATE,
    MAX_LENGTH,
    TEMPERATURE,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAINING_METRICS,
    TRAIN_PROCESSED,
    ensure_dirs,
)
from data_utils import load_jsonl


def build_examples() -> list[InputExample]:
    examples: list[InputExample] = []
    for row in load_jsonl(TRAIN_PROCESSED):
        question = row.get("body", "")
        snippets = row.get("snippets", []) or []
        if snippets:
            for snippet in snippets:
                text = str(snippet.get("text", "")).strip()
                if text:
                    examples.append(InputExample(texts=[question, text]))
        else:
            for document in row.get("document_contents", []) or []:
                text = str(document.get("text", "")).strip()
                if text:
                    examples.append(InputExample(texts=[question, text]))
                    break
    if not examples:
        raise ValueError("No training examples found. Run prepare_data.py first.")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune PubMedBERT retriever with InfoNCE.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    ensure_dirs()
    model = SentenceTransformer(BASE_RETRIEVER_MODEL, device=args.device)
    model.max_seq_length = MAX_LENGTH
    train_loss = losses.MultipleNegativesRankingLoss(model=model, scale=1.0 / TEMPERATURE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    examples = build_examples()

    history: list[dict[str, float | int]] = []
    step_history: list[dict[str, float | int]] = []
    global_step = 0
    for epoch in range(args.epochs):
        random.Random(42 + epoch).shuffle(examples)
        dataloader = DataLoader(
            examples,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=lambda batch: batch,
        )
        epoch_loss = 0.0
        step_count = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            sentence_features, labels = model.smart_batching_collate(batch)
            sentence_features = [
                batch_to_device(feature, args.device) for feature in sentence_features
            ]
            labels = labels.to(args.device)
            loss_value = train_loss(sentence_features, labels)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            value = float(loss_value.detach().cpu().item())
            global_step += 1
            step_count += 1
            epoch_loss += value
            step_history.append(
                {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": value,
                }
            )
            progress.set_postfix(loss=f"{value:.4f}")
        history.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss / max(step_count, 1),
                "steps": step_count,
            }
        )

    FINETUNED_RETRIEVER_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(FINETUNED_RETRIEVER_DIR))
    metrics = {
        "base_model": BASE_RETRIEVER_MODEL,
        "output_model": str(FINETUNED_RETRIEVER_DIR),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "temperature": TEMPERATURE,
        "example_count": len(examples),
        "history": history,
        "step_history": step_history,
    }
    TRAINING_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
