"""Merge a pre-trained LoRA adapter into its base model on CPU.

Runs as a CPU-only SageMaker training job (see 2b-import-pretrained.ipynb). It writes the
same files to /opt/ml/model that scripts/train.py writes after a GPU run with
merge_weights=true: bf16 safetensors shards (max 2 GB each) + index, config.json,
generation_config.json and the base model's tokenizer (tokenizer.json,
tokenizer_config.json, chat_template.jinja). 3-deployment.ipynb can then deploy it
unchanged.
"""

import argparse
import json
import os
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", default=os.environ.get("SM_CHANNEL_ADAPTER", "/opt/ml/input/data/adapter"))
    parser.add_argument("--output_dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    # Defaults to the base model recorded in adapter_config.json
    parser.add_argument("--model_id", default=None)
    args = parser.parse_args()

    with open(os.path.join(args.adapter_dir, "adapter_config.json")) as f:
        model_id = args.model_id or json.load(f)["base_model_name_or_path"]
    print(f"base model: {model_id}\nadapter: {args.adapter_dir}\noutput: {args.output_dir}", flush=True)

    start = time.time()
    # Same dtype and loading options train.py uses for its merge
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    print(f"base model loaded in {time.time() - start:.0f}s", flush=True)

    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model = model.merge_and_unload()
    print(f"adapter merged in {time.time() - start:.0f}s", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="2GB")

    # train.py saves the base model's tokenizer (pad_token falls back to eos_token)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(args.output_dir)

    for name in sorted(os.listdir(args.output_dir)):
        size = os.path.getsize(os.path.join(args.output_dir, name))
        print(f"{size / 1e9:8.3f} GB  {name}", flush=True)
    print(f"done in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
