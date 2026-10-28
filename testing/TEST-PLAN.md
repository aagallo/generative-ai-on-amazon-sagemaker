# Test plan for the workshop account (branch `test/combined`, not part of any PR)

```bash
git clone -b test/combined https://github.com/aagallo/generative-ai-on-amazon-sagemaker.git
cd generative-ai-on-amazon-sagemaker
python testing/check_quotas.py <region>      # 0 = that type can't be used; fallback lists drop those
```

Clear notebook outputs before `git pull` (Edit > Clear Outputs of All Cells), or pulls stop on modified notebooks.

## 1. CPU track (no GPU) — `examples/cpu-lora-quickstart/`

`01-sft.ipynb` first, then `02-continued-pretraining`, `03-dpo`, `04-grpo` (independent; can run in parallel).
Measured in us-east-1: 02 CPT 5 min (questions 0% → 100%), 03 DPO 9.4 min (DPO+SFT exact 0% → 90%; plain DPO
drops JSON, by design of the lesson), 04 GRPO 13.5 min (exact 0% → 40%, reward 0.46 → 0.79).

### 01-sft.ipynb (was quickstart.ipynb)

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
* Tested end to end on SageMaker (us-east-1): 1 → 2b → 3 (eval job waited 79 min for a g6e GPU, then ran in 7.8
  billed min: L1 0.220 vs base 0.478 vs mean-action 0.359, Approved) → 4 (endpoint in service in 8 min, ~0.45 s per
  request) → 5 (cleanup). The first request took 86 s; the handler now warms up during load to avoid that

## Cleanup

OpenVLA: `5-cleanup.ipynb`. Workshop: the last cells of each 3-deployment notebook. Quickstart: its last cell.
