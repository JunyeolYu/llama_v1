# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded

    def eval(
        self,
        prompts: List[str],
    ) -> List[str]:
        bsz = len(prompts)
        # assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [prompt[2]+prompt[4][:-1] for prompt in prompts] # JUNYEOL: concat prompt and ending, delete last token
        # max_prompt_size = max([len(t) for t in prompt_tokens])
        prompt_size = [prompt[1]+prompt[-1]-1 for prompt in prompts]
        # print(prompt_tokens)
        tokens = torch.full((bsz, max(prompt_size)), 0, device='cuda')
        # print(tokens.shape)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : prompt_size[k]] = torch.tensor(t)

        with torch.no_grad():
            multi_logits = torch.nn.functional.log_softmax(self.model.forward(tokens, 0), dim=-1)#.cpu()

        ### JUNYEOL ###
        # input_ = [prompt[1]-1 for prompt in prompts]
        # ending_ = [prompt[4] for prompt in prompts]
        # endlen = [prompt[-1] for prompt in prompts]
        
        res = []
        for logits, prompt in zip(multi_logits, prompts):
            #logits, input, ending, el in zip(multi_logits, input_, ending_, endlen):
            input, ending, el = prompt[1]-1, prompt[4], prompt[-1]
            logits = logits[input:input+el].unsqueeze(0)  # [1, seq, vocab]
            ending = torch.tensor(ending, dtype=torch.long, device='cuda').view(1,-1,1)#.unsqueeze(0).unsqueeze(-1)#.cpu()
            answer = torch.gather(logits, 2, ending).squeeze(-1).sum()  # [1, seq]
            res.append(answer)

        for prompt, ans in zip(prompts, res):
            prompt[3] = ans
        return prompts
        
def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
