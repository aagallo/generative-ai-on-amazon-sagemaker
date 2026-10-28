"""Prints the SageMaker quotas for every instance type the labs target, and flags the ones at 0.

Run in a Studio notebook cell (or locally with AWS credentials): python check_quotas.py [region]
Needs servicequotas:ListServiceQuotas. Quotas are per account and per region.
"""
import sys

import boto3

TARGETS = {
    "CPU fine-tuning techniques": {
        "training job usage": ["ml.c7i.4xlarge", "ml.m7i.4xlarge", "ml.c6i.4xlarge", "ml.m6i.4xlarge", "ml.m5.4xlarge"],
        "spot training job usage": ["ml.c7i.4xlarge"],
        "endpoint usage": ["ml.m5.xlarge", "ml.m7i.xlarge", "ml.c7i.2xlarge"],
    },
    "OpenVLA (examples/openvla-lora-sagemaker)": {
        "training job usage": ["ml.g6e.xlarge", "ml.g6e.2xlarge", "ml.g6e.4xlarge", "ml.g6e.8xlarge",
                               "ml.g6e.12xlarge", "ml.g6e.24xlarge"],
        "spot training job usage": ["ml.g6e.xlarge", "ml.g6e.12xlarge"],
        "endpoint usage": ["ml.g6.2xlarge", "ml.g5.2xlarge", "ml.g6e.xlarge", "ml.g5.4xlarge", "ml.g6e.2xlarge"],
    },
    "Distributed training workshop, options 1-3 (CPT/SFT/DPO)": {
        "training job usage": ["ml.g5.12xlarge", "ml.g6.12xlarge", "ml.g6e.12xlarge", "ml.g5.24xlarge",
                               "ml.m5.4xlarge"],  # m5.4xlarge: 2b CPU merge job
        "spot training job usage": ["ml.g5.12xlarge"],
        "endpoint usage": ["ml.g5.2xlarge", "ml.g6.2xlarge", "ml.g6e.xlarge", "ml.g5.4xlarge", "ml.g6.4xlarge"],
    },
    "Distributed training workshop, option 4 (GRPO)": {
        "training job usage": ["ml.g6e.12xlarge", "ml.g6e.24xlarge"],
        "spot training job usage": ["ml.g6e.12xlarge"],
        "endpoint usage": ["ml.g6e.2xlarge", "ml.g6e.4xlarge", "ml.g6e.8xlarge"],
    },
}


def main(region=None):
    sq = boto3.client("service-quotas", region_name=region)
    quotas = {q["QuotaName"]: q["Value"]
              for page in sq.get_paginator("list_service_quotas").paginate(ServiceCode="sagemaker")
              for q in page["Quotas"]}
    print(f"SageMaker quotas in {sq.meta.region_name} (0 = cannot launch; request an increase in Service Quotas)\n")
    for lab, usages in TARGETS.items():
        print(f"== {lab}")
        for usage, types in usages.items():
            cells = []
            for t in types:
                v = quotas.get(f"{t} for {usage}")
                cells.append(f"{t}={'n/a' if v is None else int(v)}{' <-- 0' if not v else ''}")
            print(f"  {usage:24} " + ", ".join(cells))
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
