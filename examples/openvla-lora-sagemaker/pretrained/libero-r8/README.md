# Pre-trained LoRA adapter: OpenVLA-7B on LIBERO (r=8)

Used by `2b-import-pretrained.ipynb` so the evaluation and deployment labs run without a training GPU. Only the
adapter is shipped; the evaluation job and the endpoint download the base model `openvla/openvla-7b` from the
Hugging Face Hub and merge the adapter into it when they start (`scripts/openvla_utils.load_openvla`).

| File | What |
|---|---|
| `adapter_config.json`, `adapter_model.safetensors` | LoRA adapter (PEFT), r=8, alpha=8, on the Llama-2 projections (q/k/v/o/gate/up/down) |
| `action_stats.json` | action normalization bounds (q01/q99) of the training split; the endpoint uses them to un-normalize |
| `eval_results.json` | base vs fine-tuned vs mean-action baseline on held-out LIBERO demonstrations |
| `MANIFEST.json` | how it was built: commit, recipe, dataset, GPU, package versions, stage timings |

How it was trained: `recipes/openvla_lora.yaml` (3 epochs, global batch 32, lr 5e-4) on `sample_data/libero-120`
(one held-out demonstration per task for validation), on a single NVIDIA RTX PRO 6000 (EC2 g7e.2xlarge Spot) with the
same `train_openvla_lora.py` the SageMaker training job runs. See `MANIFEST.json` for the exact commit.

## Licenses and attribution

* **Base model:** [openvla/openvla-7b](https://huggingface.co/openvla/openvla-7b) (MIT). OpenVLA is built on
  Llama 2: *Llama 2 is licensed under the LLAMA 2 Community License, Copyright (c) Meta Platforms, Inc. All Rights
  Reserved.* Use of this adapter together with the base model is subject to the
  [Llama 2 Community License](https://ai.meta.com/llama/license/) and its
  [Acceptable Use Policy](https://ai.meta.com/llama/use-policy/).
* **Training data:** [physical-intelligence/libero](https://huggingface.co/datasets/physical-intelligence/libero)
  (CC BY 4.0), a LeRobot-format copy of the [LIBERO](https://libero-project.github.io/) benchmark
  (Liu et al., 2023).
