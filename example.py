# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama import ModelArgs, Transformer, Tokenizer, LLaMA

from tqdm import tqdm
import datasets
import numpy as np

import re


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

# Request Id, Length, InputData, LogProb, Ending Length, Normalized Log Prob, label
class RequestInstance:
    def __init__(self, request_id, activity_label, context, endings_, tokenizer, label):
        self.request_id = request_id
        self.activity_label = activity_label
        self.context = context
        self.endings = []
        for i in range(4):
            self.endings.append(tokenizer.encode(self.preprocess(endings_[i]), bos=True, eos=False)[1:])

        self.tokenizer = tokenizer
        self.label = label
        self.requests = self.build_requests()

    def preprocess(self,text):
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    def build_requests(self):
        self.context = self.tokenizer.encode(self.preprocess(self.activity_label) + self.preprocess(": ") + self.preprocess(self.context), bos=True, eos=False)[1:] # delete <s> token
        return [
            [self.request_id,len(self.context), self.context, 0.0, ending_tok, self.label, i, len(ending_tok)] for i,ending_tok in enumerate(self.endings)
        ]

def load_hellaswag():
    hellaswag = datasets.load_dataset('hellaswag')
    validation = hellaswag['validation']
    validation_zeroshot = validation.filter(lambda example: example['split_type'] == 'zeroshot')
    print("Hellaswag dataset load finish , len: " + str(len(validation_zeroshot)))
    return validation_zeroshot

def engineering_dataset(validation_zeroshot, tokenizer, max_batch_size):
    requests = []
    for i, row in tqdm(enumerate(validation_zeroshot)):
        temp = RequestInstance(i, row['activity_label'], row['ctx'], row['endings'], tokenizer, int(row['label']))
        requests.extend(temp.requests)

    final_reqs = []
    for i in range(0, len(requests), max_batch_size):
        final_reqs.append(requests[i:i+max_batch_size])

    return final_reqs

def calculate_accuracy(res):
    acc = 0
    nacc = 0

    for r in range(0,len(res), 4):
        try:
            outs = sorted(res[r:r+4], key=lambda x: x[6])
            # assert that outs order is correct
            assert outs[0][6] == 0
            assert outs[1][6] == 1
            assert outs[2][6] == 2
            assert outs[3][6] == 3

            # [self.request_id,len(self.context), self.context, 0.0, ending_tok, self.label, i]
            logs = [out[3] for out in outs]

            ending_lens = [len(out[4]) for out in outs]
            nlogs = [log/ending_lens[i] for i,log in enumerate(logs)]

            pred_label = logs.index(max(logs))
            norm_pred_label = nlogs.index(max(nlogs))

            label = outs[0][5]

            if pred_label == label:
                acc += 1
            if norm_pred_label == label:
                nacc += 1
        except:
            print("Failed while calculating accuracy")
            pass
    
    total_len = len(res)/4
    acc = acc/total_len
    nacc = nacc/total_len
    print("Accuracy:", acc)
    print("Normalized Accuracy:", nacc)

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 170,
    max_batch_size: int = 32,
):
    start_main = time.time()
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    # Prepare the dataset
    start_dataset = time.time()
    validation_zeroshot = []
    final_reqs = []
    validation_zeroshot = load_hellaswag()
    tokenizer = Tokenizer(model_path=tokenizer_path)
    final_reqs = engineering_dataset(validation_zeroshot, tokenizer, max_batch_size)

    start_model = time.time()
    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    start_eval = time.time()
    res = []
    for i in tqdm(range(len(final_reqs))):
        results = generator.eval(final_reqs[i])
        res.extend(results)

    start_cal = time.time()
    res = sorted(res, key=lambda x: x[0])
    calculate_accuracy(res)

    end = time.time()

    t_total = end - start_main
    t_before = start_dataset - start_main
    t_dataset = start_model - start_dataset
    t_load = start_eval - start_model
    t_eval = start_cal - start_eval
    t_cal = end - start_cal

    print(f"Total_time    : {t_total} s")
    print(f"Preprocessing : {t_dataset} s")
    print(f"Model_loading : {t_load} s")
    print(f"Evaluation    : {t_eval} s")
    print(f"Acc_Cal       : {t_cal} s")
    print(f"Others        : {t_before} s")
    
if __name__ == "__main__":
    fire.Fire(main)
