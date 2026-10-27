# Pre-trained LoRA adapter: supervised fine-tuning (SFT)

A LoRA adapter (r=16, alpha=32, all attention and MLP projections) for `Qwen/Qwen3-4B-Instruct-2507`, trained ahead of time so you can run this option without GPU capacity. [2b-import-pretrained.ipynb](../2b-import-pretrained.ipynb) merges it into the base model on a CPU SageMaker training job; [3-deployment.ipynb](../3-deployment.ipynb) then deploys it like a Lab 2 result.

| File | Content |
|---|---|
| `adapter/adapter_config.json`, `adapter/adapter_model.safetensors` | PEFT LoRA adapter, BF16, about 66 MB |
| `MANIFEST.json` | How the adapter was produced (commit, instance, package versions, timings) |
| `all_results.json`, `train_results.json`, `trainer_state.json` | Trainer metrics and the per-step loss log |

## How it was trained

* Code: `scripts/train.py` and the args of [2-supervised-fine-tuning.ipynb](../2-supervised-fine-tuning.ipynb), from https://github.com/aagallo/generative-ai-on-amazon-sagemaker.git commit `6cf538c43b7e7fbd999303ce1dfcc70b6758b35f` (branch `test/combined`).
* Dataset: FreedomIntelligence/medical-o1-reasoning-SFT en (first 1000 rows: 891 train / 100 val / 9 test), prepared as in [1-prepare-data.ipynb](../1-prepare-data.ipynb).
* Recipe: QLoRA (4-bit base), bf16, learning rate 1.2e-4, 4 epochs, max_length 4096, same as Lab 2.
* Hardware: EC2 `g7e.2xlarge` (1x NVIDIA RTX PRO 6000 Blackwell Server Edition), `torchrun --nproc_per_node 1`, FSDP on.
* Batch: per-device batch 2 x gradient accumulation 8 x 1 GPU = global batch 16, matching the workshop's 2 x 2 x 4 GPUs on `ml.g5.12xlarge`.
* Attention: `sdpa` instead of the workshop's `flash_attention_2` (same math, different kernel).
* Packages: torch 2.8.0+cu128, transformers 5.13.1, peft 0.19.1, trl 1.5.1.
* Result: train loss 1.3159 after 4 epoch(s) on 891 samples (820 s).

## License

* Base model `Qwen/Qwen3-4B-Instruct-2507`: Apache-2.0.
* Training data `FreedomIntelligence/medical-o1-reasoning-SFT`: Apache-2.0 (per its Hugging Face dataset card).

The adapter is a derivative of both; follow these licenses (including attribution for the dataset) when you use or redistribute it.
