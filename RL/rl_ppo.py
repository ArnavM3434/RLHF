
import os
import math
import json
from pathlib import Path
from typing import Dict, Any
import glob
import shutil
import wandb
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from peft import LoraConfig, get_peft_model, PeftModel

from torch.utils.tensorboard import SummaryWriter

from transformers import get_linear_schedule_with_warmup

from trl import create_reference_model


wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead

base_model = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

policy_model = AutoModelForCausalLMWithValueHead.from_pretrained("ArnavM3434/gpt2-alpaca-second-try")

policy_model.to(device)
test_prompts = [
    "What is machine learning?",
    "Explain reinforcement learning",
    "How does PPO work?",
]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = policy_model.generate(
        **inputs,
        max_new_tokens=512,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        do_sample=False,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")
    print("-" * 80)

ref_model = create_reference_model(policy_model)

for param in ref_model.parameters():
    param.requires_grad = False

def inspect_trainable_params(model):
    total = 0
    trainable = 0
    details = []
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            details.append(n)
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    print("Example trainable params:", details[:20])
    return details

inspect_trainable_params(policy_model)

def make_trainable(model):
  for name, param in model.named_parameters():
      if "lora_" in name or "v_head" in name:
          param.requires_grad = True
      else:
          param.requires_grad = False

make_trainable(policy_model)

inspect_trainable_params(policy_model)

#Load Reward Model

reward_model = AutoModelForSequenceClassification.from_pretrained("OpenAssistant/reward-model-deberta-v3-base")
reward_tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/reward-model-deberta-v3-base")

inspect_trainable_params(reward_model)

reward_model.to(device)

reward_model.eval()

for param in reward_model.parameters():
    param.requires_grad = False

#Prompt Dataset

from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca", split="train")

dataset[0]

dataset

dataset = dataset.remove_columns(["output", "text"])

dataset[0]

def tokenize(sample):
    sample["query"] = f"Human: {sample['instruction']} {sample['input']} Assistant: "
    sample["input_ids"] = tokenizer.encode(sample["query"], padding = "max_length", truncation = True, max_length = 128)
    return sample

tokenized_dataset = dataset.map(
    tokenize,
    batched=False,
    remove_columns = ["instruction", "input"]
)

tokenized_dataset[0]['query']

#PPO

from trl import PPOTrainer, PPOConfig
import numpy as np

config = PPOConfig(
    model_name="gpt2",
    learning_rate=5e-7,
    batch_size=4,
    mini_batch_size=2,
    gradient_accumulation_steps=1,
    log_with="wandb",
    target_kl=2.0,
    cliprange=0.05,
    cliprange_value=0.1,
    vf_coef=0.1,
    init_kl_coef=0.5,
    adap_kl_ctrl=True,
    gamma=0.99,
    lam=0.95,
    ppo_epochs=2
)

ppo_trainer = PPOTrainer(
    model=policy_model,
    config=config,
    dataset=tokenized_dataset,
    tokenizer=tokenizer,
    ref_model = ref_model
)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

save_dir = "./ppo-model"
checkpoint_prefix = "checkpoint"

def save_training_state():
    save_dir = Path(save_dir)
    ckpt_dir = save_dir / f"{checkpoint_prefix}"
    ensure_dir(ckpt_dir)

    model_to_save = policy_model
    peft_save_dir = ckpt_dir / "adapter"
    model_to_save.save_pretrained(peft_save_dir)

    print(f"Saved checkpoint to {ckpt_dir}")

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 128, # Reduced to prevent OOM,
    "temperature": 0.8,  # Add temperature for more stable sampling
    "repetition_penalty": 1.2  # Prevent repetition
}

import warnings
import logging

warnings.filterwarnings("ignore", message=".*right-padding was detected.*")

from tqdm import tqdm
from torch.cuda.amp import autocast

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):

    #Fix the shape, it should be a list of 4 tensors, each (128, )
    query_tensors_batch = batch["input_ids"]
    query_tensor_2d = torch.stack(query_tensors_batch)
    query_tensor_correct = query_tensor_2d.transpose(0, 1)
    query_tensors = [query_tensor_correct[i] for i in range(query_tensor_correct.size(0))]

    # Step 2: Remove padding tokens from queries before generation
    unpadded_queries = []
    for i, query in enumerate(query_tensors):
        non_pad_mask = query != tokenizer.pad_token_id
        unpadded_query = query[non_pad_mask]
        unpadded_queries.append(unpadded_query)
        #print(f"Query {i}: padded_length={len(query)}, actual_length={len(unpadded_query)}")

    response_tensors_full = ppo_trainer.generate(unpadded_queries, **generation_kwargs)

    # Step 4: Extract only the response tokens
    response_only_tensors = []
    for i, full_response in enumerate(response_tensors_full):
        query_length = len(unpadded_queries[i])
        response_length = len(full_response) - query_length

        #print(f"Sequence {i}: query_length={query_length}, full_response_length={len(full_response)}, response_length={response_length}")

        if response_length > 0:
            response_only = full_response[query_length:]
        else:
            # Fallback if something goes wrong
            print("Something is wrong, the response length is 0")
            response_only = torch.tensor([tokenizer.eos_token_id], device=full_response.device)

        response_only_tensors.append(response_only)

    # Step 5: Decode and verify responses
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_only_tensors]

    # # Print actual responses to verify they're not just EOS
    # for i, (query, resp) in enumerate(zip(batch["query"], batch["response"])):
    #     print(f"Response {i}: '{resp}' (query: '{query[:50]}...')")

    # Step 6: Compute rewards
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    with torch.no_grad():
        reward_inputs = reward_tokenizer(
            texts, padding=True, truncation=True,
            return_tensors="pt", max_length=384
        ).to(device)

        rewards = reward_model(**reward_inputs).logits.squeeze(-1)
        if torch.isnan(rewards).any():
          rewards = torch.nan_to_num(rewards, nan=0.0)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        rewards = rewards.clamp(-3, 3)
        rewards = [r for r in rewards]

    # Final check
    #print(f"Final - Response lengths: {[len(r) for r in response_only_tensors]}")

    # Use the original padded queries for PPO step (as expected by the trainer)
    stats = ppo_trainer.step(query_tensors, response_only_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

ppo_trainer.save_model("ppo_model")
save_training_state()