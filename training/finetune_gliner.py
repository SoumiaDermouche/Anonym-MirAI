"""Fine-tuning de GLiNER (urchade/gliner_multi-v2.1) sur le subset français
d'ai4privacy/pii-masking-openpii-1m, préparé par prepare_gliner_data.py.

Usage :
    # essai rapide sur un petit échantillon (sanity check du pipeline)
    python training/finetune_gliner.py --train data/gliner_train_fr_sample.json \
        --eval data/gliner_eval_fr_sample.json --output-dir models/gliner-fr-poc \
        --max-steps 60 --batch-size 4

    # entraînement complet
    python training/finetune_gliner.py --train data/gliner_train_fr_full.json \
        --eval data/gliner_eval_fr.json --output-dir models/gliner-fr-full \
        --num-epochs 3 --batch-size 8
"""
import argparse
import json
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
from gliner import GLiNER

BASE_MODEL = "urchade/gliner_multi-v2.1"


def load_json(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--train", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--num-epochs", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=1)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--eval-steps", type=int, default=None)
    args = p.parse_args()

    train_dataset = load_json(args.train)
    eval_dataset = load_json(args.eval)
    print(f"train: {len(train_dataset)} exemples | eval: {len(eval_dataset)} exemples")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Chargement du modèle de base {args.model} sur {device}...")
    model = GLiNER.from_pretrained(args.model)
    model = model.to(device)

    training_kwargs = dict(
        learning_rate=args.lr,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=True,  # disque limité : pas besoin de reprendre l'optimizer (2x la taille du modèle)
        logging_steps=args.logging_steps,
        use_cpu=(device == "cpu"),
        bf16=False,
        dataloader_num_workers=2,
        report_to="none",
    )
    if args.num_epochs is not None:
        training_kwargs["num_train_epochs"] = args.num_epochs
        training_kwargs["max_steps"] = -1
    else:
        training_kwargs["max_steps"] = args.max_steps

    training_kwargs["eval_strategy"] = "steps"
    training_kwargs["eval_steps"] = args.eval_steps or args.save_steps

    training_args = model.create_training_args(output_dir=args.output_dir, **training_kwargs)

    trainer = model.train_model(train_dataset, eval_dataset, training_args=training_args)

    trainer.save_model(args.output_dir)
    print(f"Modèle fine-tuné sauvegardé dans {args.output_dir}")


if __name__ == "__main__":
    main()
