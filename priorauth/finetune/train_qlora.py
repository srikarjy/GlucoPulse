"""
QLoRA supervised fine-tuning of a document-judgement model.

    python -m priorauth.finetune.train_qlora --config priorauth/finetune/config.yaml            # needs a CUDA GPU
    python -m priorauth.finetune.train_qlora --config priorauth/finetune/config.yaml --dry-run  # anywhere: validates data + config

STATUS: the dry-run and dataset are tested in CI. The training loop below has NOT been executed in this repo's
environment (macOS, no CUDA); it follows the documented peft/trl QLoRA recipe. Treat the first GPU run as the test,
and do not quote any model-quality number until `evaluate_adapter.py` has been run on the held-out splits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from priorauth.finetune.dataset import fingerprint, write

REQUIRED = {"base_model", "output_dir", "data_dir", "max_seq_len", "seed", "quant", "lora", "train"}


def load_config(path: str) -> dict:
    cfg = yaml.safe_load(open(path))
    missing = REQUIRED - set(cfg)
    if missing:
        raise ValueError(f"config missing keys: {sorted(missing)}")
    if not cfg["quant"].get("load_in_4bit"):
        raise ValueError("QLoRA requires quant.load_in_4bit: true")
    if cfg["lora"]["r"] <= 0 or cfg["train"]["learning_rate"] <= 0:
        raise ValueError("lora.r and train.learning_rate must be positive")
    return cfg


def prepare_data(cfg: dict) -> dict[str, int]:
    counts = write(cfg["data_dir"])
    d = Path(cfg["data_dir"])
    tr = {fingerprint(json.loads(l)) for l in (d / "train.jsonl").open()} | {fingerprint(json.loads(l)) for l in (d / "val.jsonl").open()}
    for held in ("test", "test_large"):
        leaked = sum(fingerprint(json.loads(l)) in tr for l in (d / f"{held}.jsonl").open())
        if leaked:
            raise RuntimeError(f"{leaked} {held} prompts also appear in train/val: refusing to train (data leakage)")
    return counts


def train(cfg: dict) -> None:  # pragma: no cover - requires CUDA + the train requirements
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    q, t, l = cfg["quant"], cfg["train"], cfg["lora"]
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
                             bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
                             bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]))
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=bnb, device_map="auto")
    ds = load_dataset("json", data_files={"train": f"{cfg['data_dir']}/train.jsonl", "validation": f"{cfg['data_dir']}/val.jsonl"})
    ds = ds.remove_columns("meta")
    peft_cfg = LoraConfig(r=l["r"], lora_alpha=l["alpha"], lora_dropout=l["dropout"], target_modules=l["target_modules"],
                          task_type="CAUSAL_LM", bias="none")
    args = SFTConfig(output_dir=cfg["output_dir"], num_train_epochs=t["epochs"], per_device_train_batch_size=t["per_device_batch_size"],
                     gradient_accumulation_steps=t["grad_accum"], learning_rate=t["learning_rate"], lr_scheduler_type=t["lr_scheduler"],
                     warmup_ratio=t["warmup_ratio"], weight_decay=t["weight_decay"], gradient_checkpointing=t["gradient_checkpointing"],
                     bf16=t["bf16"], logging_steps=t["logging_steps"], eval_strategy="steps", eval_steps=t["eval_steps"],
                     save_steps=t["save_steps"], max_length=cfg["max_seq_len"], seed=cfg["seed"], assistant_only_loss=True,
                     report_to="none")
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"],
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(cfg["output_dir"])           # LoRA adapter only (tens of MB), not the 7B base


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    counts = prepare_data(cfg)
    print("dataset:", counts, "| no train/test prompt overlap")
    if a.dry_run:
        eff = cfg["train"]["per_device_batch_size"] * cfg["train"]["grad_accum"]
        print(f"dry-run OK: {counts['train']} examples, effective batch {eff}, ~{counts['train'] * cfg['train']['epochs'] // eff} optimizer steps")
        return
    train(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
