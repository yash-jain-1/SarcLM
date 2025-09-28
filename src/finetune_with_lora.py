"""
finetune_after_script_b.py

Uses robust_llava_loader.load_model_and_tokenizer() (script B output) to load the model,
auto-detects good LoRA target_modules, and runs PEFT (LoRA) fine-tuning with trl.SFTTrainer.

Usage: python finetune_after_script_b.py
"""

import os
import re
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel

# try to import helper for preparing k-bit training (peft versions vary)
try:
    from peft import prepare_model_for_kbit_training
except Exception:
    try:
        from peft.utils import prepare_model_for_kbit_training
    except Exception:
        prepare_model_for_kbit_training = None

# robust loader from script B
from robust_llava_loader import load_model_and_tokenizer

# --------------------- USER CONFIG ---------------------
BASE_MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
DATASET_PATH = "meld_with_rationales.jsonl"   # jsonl containing utterance, sentiment, rationale
OUTPUT_DIR = "./llava-peft-adapters-auto"
USE_4BIT_IF_AVAILABLE = True
MAX_SEQ_LENGTH = 512
PER_DEVICE_BATCH_SIZE = 4
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
GRADIENT_ACCUMULATION_STEPS = 1
SAVE_STEPS = 200
LOGGING_STEPS = 20
# -------------------------------------------------------

# helper: create the training prompt
def build_prompt(example):
    return (
        "You are a sentiment analysis expert. Analyze the following utterance and provide "
        "the sentiment along with a step-by-step rationale for your decision.\n\n"
        "### Utterance:\n"
        f"{example.get('utterance','')}\n\n"
        "### Analysis:\n"
        f"Sentiment: {example.get('sentiment','')}\n"
        f"Rationale: {example.get('rationale','')}"
    )

# helper: scan model.named_modules() and choose candidate target module name substrings
def auto_detect_target_module_names(model, prefer_text=True):
    """
    Returns a list of module-name substrings to use in LoraConfig.target_modules.
    Strategy:
      - Collect names of submodules that look like projections (q_proj, k_proj, v_proj, out_proj, o_proj)
      - Prefer modules under 'model' that contain tokens like 'self_attn', 'attn', 'q_proj' etc.
      - If prefer_text=True, try to exclude modules under vision tower (module name containing 'vision' or 'vision_tower')
    """
    proj_patterns = set()
    name_list = [n for n, _ in model.named_modules()]

    for n in name_list:
        # skip top-level empty name
        if not n:
            continue
        # skip vision modules if preferring text modules
        if prefer_text and ("vision" in n or "vision_tower" in n or "vision_model" in n):
            continue
        # find typical projection/fc names in module path
        if re.search(r"(q_proj|k_proj|v_proj|o_proj|out_proj|gate_proj|up_proj|down_proj|fc1|fc2|mlp)", n):
            # extract final token (last part after '.')
            final = n.split(".")[-1]
            proj_patterns.add(final)
    # fallback if empty
    if not proj_patterns:
        # default common names
        proj_patterns = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    # keep consistent ordering and return as list
    return sorted(list(proj_patterns))

def format_and_map(example, tokenizer):
    text = build_prompt(example)
    eos = tokenizer.eos_token or ""
    return {"text": text + eos}

def main():
    # decide 4-bit usage
    use_4bit = False
    if USE_4BIT_IF_AVAILABLE:
        try:
            import bitsandbytes  # noqa: F401
            use_4bit = True
        except Exception:
            print("bitsandbytes not installed/found — running without 4-bit.")

    # 1) Load model + tokenizer (robust loader)
    print("Loading model + tokenizer (robust loader)...")
    model, tokenizer = load_model_and_tokenizer(model_name=BASE_MODEL_NAME, use_4bit=use_4bit)
    print("Loaded model and tokenizer. Model dtype hint:", getattr(model, "dtype", None))
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    # 2) Auto-detect target_modules for LoRA (based on model module names)
    print("Auto-detecting candidate LoRA target module name tokens from model.named_modules()...")
    detected = auto_detect_target_module_names(model, prefer_text=True)
    print("Detected target-module name tokens (candidates):", detected)

    # We'll use these tokens as LoraConfig.target_modules (PEFT expects substrings)
    target_modules = detected

    # 3) Prepare model for k-bit training (if using 4-bit & helper present)
    if use_4bit:
        if prepare_model_for_kbit_training is not None:
            print("Preparing model for k-bit training (peft.prepare_model_for_kbit_training)...")
            model = prepare_model_for_kbit_training(model)
        else:
            print("prepare_model_for_kbit_training not available in this peft version — continuing.")

    # 4) Create LoraConfig and wrap model
    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    print("Applying LoRA with LoraConfig:", lora_cfg)
    model = get_peft_model(model, lora_cfg)
    print("PEFT/LoRA applied. Peft model keys:", list(model.named_parameters())[:5])

    # 5) Load and format dataset
    print("Loading dataset from", DATASET_PATH)
    ds = load_dataset("json", data_files=DATASET_PATH, split="train")
    print("Dataset size:", len(ds))
    # map to `text` field expected by SFTTrainer
    ds = ds.map(lambda ex: format_and_map(ex, tokenizer), remove_columns=ds.column_names)

    # 6) TrainingArguments + trainer
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=use_4bit or (torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 12 * 1024 ** 2),
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        save_total_limit=3,
        report_to="none",
    )

    # import here to avoid top-level dependency until needed
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        peft_config=lora_cfg,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    # 7) Dry-run: single step to validate forward/backward
    print("Running a 1-step dry-run to validate training loop...")
    try:
        trainer.train(max_steps=1)
        print("Dry-run succeeded.")
    except Exception as e:
        print("Dry-run failed — inspect traceback. Error:", e)
        raise

    # 8) Full training
    print("Starting full training...")
    trainer.train()
    print("Training finished.")

    # 9) Save PEFT adapters
    print("Saving adapters to:", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    print("Saved. You can load later with PeftModel.from_pretrained(base_model, OUTPUT_DIR)")

if __name__ == "__main__":
    main()
