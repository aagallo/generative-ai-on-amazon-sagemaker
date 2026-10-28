# Test plan for the workshop account (branch `test/combined`, not part of any PR)

```bash
git clone -b test/combined https://github.com/aagallo/generative-ai-on-amazon-sagemaker.git
cd generative-ai-on-amazon-sagemaker
python testing/check_quotas.py <region>      # 0 = that type can't be used; fallback lists drop those
```

Clear notebook outputs before `git pull` (Edit > Clear Outputs of All Cells), or pulls stop on modified notebooks.

## 1. CPU quickstart (no GPU) — `examples/cpu-lora-quickstart/quickstart.ipynb`

Run all cells. Expected (measured in us-east-1):
* 3 training jobs in parallel (on-demand / spot / fallback), ~4 min each; spot billed ~57 s vs ~180 s
* exact match base 0% → fine-tuned 100% for all three
* 3 endpoints (real-time / pools / serverless) in service after ~3.5 min; serverless first call ~46 s (cold start),
  then ~4 s; last cell deletes them

## 2. Distributed-training workshop, no-GPU path — `workshops/distributed-training-deployment-on-sagemaker-ai/solution-2-sagemaker-training/option-{1,2,3}-*/`

Per option: `1-prepare-data.ipynb` → **`2b-import-pretrained.ipynb`** (instead of 2) → `3-deployment.ipynb`.
* 2b: CPU merge job (`ml.m5.4xlarge`) ~5 min, writes the model where 3-deployment looks for it (tested: SFT)
* 3-deployment: GPU endpoint with 5 instance pools and a 1 h provisioning budget. Tested end to end for SFT
  (2b → 3-deployment → streamed answer → cleanup): GPU in ~10 min on the last run (~48 min on an earlier one; the
  first attempt failed after 10 min with the old 600 s budget, fixed). The SFT answer starts with a stray
  `<tool_call>` token: that comes from the workshop's QLoRA recipe, not the import
* Fixed today: vLLM container failed to load the tokenizer saved by transformers 5 (`'list' object has no
  attribute 'keys'`). Affects normal Lab 2 runs too; `train.py` and the merge script now write a compatible config
* GRPO (option 4) has no shipped adapter (veRL does a full fine-tune, 8.2 GB); use its normal Lab 2

## 3. Distributed-training workshop, GPU training path (capacity changes)

In `2-*.ipynb`, set `CAPACITY = "spot"` or `"fallback"` in the capacity cell; mlflow_uri must be the MLflow App ARN
(or empty to skip). Expect: fallback list filtered by quota; FSDP without CPU offload (fixes the grad-clipping crash).

## 4. OpenVLA — `examples/openvla-lora-sagemaker/`

`config.py`: `DATASET = "libero"` (default), `RUN_MODE = "smoke"`, `CAPACITY = "fallback"`.
* No-GPU path: `1-prepare-data` (CPU, ~15 s) → **`2b-import-pretrained`** (uploads the shipped 40 MB adapter,
  registers it; tested) → `3-evaluation` (GPU eval job; base + adapter merged at load) → `4-deployment` → `5-cleanup`
* Expected eval (notebook 3, 400 held-out frames): L1 fine-tuned ~0.22 vs base ~0.48 vs mean-action ~0.36 → Approved
* GPU training path: `2-fine-tune-vla` smoke run on a single L40S (fallback over g6e sizes)
* Not yet run on SageMaker: notebook 3 eval job and notebook 4 endpoint from the adapter (both waited for GPU
  capacity in us-east-1; the same code paths passed on an EC2 RTX PRO 6000 box)

## Cleanup

OpenVLA: `5-cleanup.ipynb`. Workshop: the last cells of each 3-deployment notebook. Quickstart: its last cell.
