# Test plan for the workshop account (branch `test/combined`, not part of any PR)

```bash
git clone -b test/combined https://github.com/aagallo/generative-ai-on-amazon-sagemaker.git
cd generative-ai-on-amazon-sagemaker
python testing/check_quotas.py <region>      # 0 = that type can't be used; fallback lists drop those
```

Clear notebook outputs before `git pull` (Edit > Clear Outputs of All Cells), or pulls stop on modified notebooks.

## 1. CPU track (no GPU) — `examples/cpu-fine-tuning-techniques/`

`01-sft.ipynb` first, then `02-continued-pretraining`, `03-dpo`, `04-grpo` (independent; can run in parallel).
Measured in us-east-1: 02 CPT 5 min (questions 0% → 100%), 03 DPO 9.4 min (DPO+SFT exact 0% → 90%; plain DPO
drops JSON, by design of the lesson), 04 GRPO 13.5 min (exact 0% → 40%, reward 0.46 → 0.79).

### 01-sft.ipynb (full workflow, ~25 min)

Run all cells. Expected (measured in us-east-1, 2026-10-28):
* setup finds or creates the account's MLflow App (first time ~5 min; `MLFLOW_APP_ARN` to reuse one, `None` to
  skip). Every job in 01-04 logs to experiment `cpu-fine-tuning-techniques`; the MLflow cell prints a link
* 1. Prepare: uploads `data/train/train.jsonl` and `data/test/test.jsonl`; the training log shows
  `data from channel: 400 train examples`
* 2. Fine-tune: 3 training jobs in parallel, ~4 min each: on-demand and fallback on `ml.c7i.4xlarge`, Spot on
  `ml.m5.4xlarge` (`c7i.4xlarge` had a Spot placement score of 1 and no Spot capacity all afternoon). If Spot
  still waits, the cell continues 10 min after on-demand and the Spot job stops itself after 30 min
* 3. Evaluate: separate eval job ~3-4 min, exact match base 0% → fine-tuned 100%; gate PASS; a new version in
  model package group `cpu-ft-sft-booking`, Approved
* 4. Deploy: 3 endpoints from the approved package in ~3.7 min; serverless first call ~30 s (cold start), then
  ~4 s; the base vs fine-tuned cell shows code-block/"7pm" answers vs bare JSON/"19:00"
* 5. Clean up deletes the endpoints (flags for the registry and S3 data are off by default)

## 2. Distributed-training workshop — `workshops/distributed-training-deployment-on-sagemaker-ai/solution-2-sagemaker-training/`

Two branches feed this section (two separate PRs):
* `feat/capacity-fallback` (PR1): in `2-*.ipynb`, the capacity cell offers `CAPACITY = "on-demand"` (default
  behaviour of the original), `"spot"` or `"fallback"`; the fallback list is filtered by training quota; Spot jobs
  checkpoint to a per-run S3 prefix and resume after an interruption. `3-deployment.ipynb` uses 5 endpoint
  instance pools with a 1 h provisioning budget.
* `fix/distributed-training`: FSDP without CPU offload in CPT/SFT (fixes the grad-clipping crash), `mlflow_uri`
  must be the MLflow App ARN (or empty to skip; a URL gives 403), and `train.py` writes a tokenizer config the vLLM
  container can load (`'list' object has no attribute 'keys'` otherwise).

Per option: `1-prepare-data` → `2-*` (try `"fallback"`, and `"spot"` if you can wait) → `3-deployment`. All of 2
and 3 need GPUs: training on `ml.g5/g6/g6e.12xlarge`-class (4 GPUs), the endpoint on one GPU. On 2026-10-28 a
DPO endpoint failed after the full 1 h with InsufficientInstanceCapacity on all 5 pools in us-east-1: capacity,
not code; retry later or in another region. (The earlier no-GPU "2b" import path was removed from PR1;
archived locally and at tag `pr1-with-2b`.)

## 3. OpenVLA — `examples/openvla-lora-sagemaker/`

`config.py`: `DATASET = "libero"` (default), `RUN_MODE = "smoke"`, `CAPACITY = "fallback"`.
* Path: `1-prepare-data` (CPU, ~15 s) → `2-fine-tune-vla` (smoke run on one L40S, fallback over g6e sizes; ~20 min
  on SageMaker) → `3-evaluation` (GPU eval job) → `4-deployment` (GPU endpoint, instance pools) → `5-cleanup`.
  Every step after 1 needs a GPU, so there is no shipped adapter / 2b path any more (removed 2026-10-28; archived
  locally and at tag `openvla-2b-import`)
* Expected eval (notebook 3, 400 held-out frames): L1 fine-tuned ~0.22 vs base ~0.48 vs mean-action ~0.36 → Approved
  (the smoke run trains less than the full recipe, so expect a smaller gain)
* Previously tested on SageMaker (us-east-1) with the shipped adapter: 3 (eval job waited 79 min for a g6e GPU, then
  ran in 7.8 billed min: L1 0.220 vs base 0.478 vs mean-action 0.359, Approved) → 4 (endpoint in service in 8 min,
  ~0.45 s per request) → 5. The first request took 86 s; the handler now warms up during load to avoid that

## Cleanup

OpenVLA: `5-cleanup.ipynb`. Workshop: the last cells of each 3-deployment notebook. Quickstart: its last cell.
