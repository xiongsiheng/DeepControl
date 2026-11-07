# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Preprocess and mix NQ + HotpotQA into one search-format parquet dataset.
Output:
  ./data/nq_hotpotqa_search/train.parquet
  ./data/nq_hotpotqa_search/test.parquet
"""

import os
import argparse

from datasets import load_dataset, concatenate_datasets

from verl.utils.hdfs_io import copy, makedirs
from deepcontrol.protocol import user_prompt


# The task prompt is the one inference sends (deepcontrol/protocol.py), so training rollouts
# start from exactly the first user turn the policy sees at test time.


def make_prefix(question: str, template_type: str) -> str:
    if template_type != "base":
        raise NotImplementedError(f"Unsupported template_type: {template_type}")
    return user_prompt(question)


def map_nq(ds, split: str, template_type: str):
    def process_fn(example, idx):
        question = (example["question"] or "").strip()
        prompt = make_prefix(question, template_type=template_type)
        solution = {"target": example["golden_answers"]}
        return {
            "data_source": "nq",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "fact-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "source": "nq",
            },
        }

    return ds.map(function=process_fn, with_indices=True)


def map_hotpotqa(ds, split: str, template_type: str):
    def process_fn(example, idx):
        question = (example["question"] or "").strip()
        prompt = make_prefix(question, template_type=template_type)
        solution = {"target": [str(example["answer"]).strip()]}
        return {
            "data_source": "hotpotqa",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "fact-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "source": "hotpotqa",
            },
        }

    return ds.map(function=process_fn, with_indices=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/nq_hotpotqa_search")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--template_type", type=str, default="base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", default=False)
    args = parser.parse_args()

    nq = load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq")
    hotpot = load_dataset("hotpotqa/hotpot_qa", "fullwiki")

    nq_train = map_nq(nq["train"], split="train", template_type=args.template_type)
    nq_test = map_nq(nq["test"], split="test", template_type=args.template_type)

    hotpot_train = map_hotpotqa(hotpot["train"], split="train", template_type=args.template_type)
    hotpot_test = map_hotpotqa(hotpot["validation"], split="test", template_type=args.template_type)

    train_dataset = concatenate_datasets([nq_train, hotpot_train])
    test_dataset = concatenate_datasets([nq_test, hotpot_test])

    if args.shuffle:
        train_dataset = train_dataset.shuffle(seed=args.seed)
        test_dataset = test_dataset.shuffle(seed=args.seed)

    os.makedirs(args.local_dir, exist_ok=True)

    print("Training set size:", len(train_dataset))
    print("Test set size:", len(test_dataset))

    train_dataset.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_dir, "test.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
