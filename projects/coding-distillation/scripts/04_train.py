"""LoRA SFT of the student on teacher data -> outputs/lora-adapter/"""
import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from distill.modeling import MODEL_ID, pick_device


def to_prompt_completion(row):
    return {"prompt": row["messages"][:-1], "completion": row["messages"][-1:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/lora-adapter")
    ap.add_argument("--max-steps", type=int, default=-1, help="cap steps (smoke test)")
    ap.add_argument("--device", help="mps or cpu (default: auto)")
    args = ap.parse_args()

    data = load_dataset("json", data_files={"train": "data/train.jsonl", "val": "data/val.jsonl"})
    data = data.map(to_prompt_completion, remove_columns=["messages"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)

    config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=3,
        max_steps=args.max_steps,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # float = ratio of total steps (transformers 5 removed warmup_ratio)
        max_length=1024,
        completion_only_loss=True,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=5,
        report_to="none",
        seed=42,
        use_cpu=pick_device(args.device) == "cpu",
    )
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"saved adapter to {args.output}")


if __name__ == "__main__":
    main()
