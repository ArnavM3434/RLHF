# RLHF Implementation (Based on InstructGPT)

I did this project to gain more familiarity with **Reinforcement Learning from Human Feedback (RLHF)** as a whole.  
It is based on the paper: [*Training language models to follow instructions with human feedback (InstructGPT)*](https://arxiv.org/abs/2203.02155).

Models pushed to the hub here:
https://huggingface.co/ArnavM3434
---

## Stages

### 1. Supervised Fine-Tuning (SFT)
- **Pretrained model:** `gpt2`  
- **LoRA configuration:** Model wrapped with a LoRA adapter (details in notebook)  
- **Dataset:** [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) — single-turn instruction–completion pairs I put in the form:

      Human: ...
      Assistant: ...

- **Training details:**
  - Loss computed on **completion only** (masked out prompts and padding tokens)
  - Training loss converged around **2.15** (started around ~2.7)
  - Results were consistent across multiple learning rates and LoRA configs  
- **Outcome:**
  - Much better **BLEU score** and **qualitative completions** than pretrained GPT-2  
  - See training runs and examples in the notebook (Note: The notebook loss starts small because this was the 3rd or 4th run from an existing checkpoint)

**Weights & Biases dashboard:**
<p align="center">
  <img src="https://i.imgur.com/Smm5Ql8.png" alt="Training loss curve" width="600"/>
</p>

---

### 2. Reward Model
- **Dataset:** `Dahoas/rm-static`  
- **Loss function:** from Bradley–Terry pairwise ranking
- **Architecture:** Added a reward head to the SFT model  
- **Observations:**
  - Training loss converged around 0.65. Validation accuracy converged around 61%
  - I modified the data to only be single turn completions to better match the distribution the SFT model was trained on, yielding better results than using the raw 'Dahoas/rm-static'

---

### 3. Proximal Policy Optimization (PPO)
- **Prompts:** Alpaca dataset  
- **Config:** See PPO configuration in notebooks.
- **Observations:**
  - KL divergence remained **negative**, despite various adjustments:
    - Reward normalization  
    - Generation tweaks  
    - Different KL coefficients
    - Different clip ranges
    - Different reward models 
  - Tried another reward model (`OpenAssistant/reward-model-deberta-v3-base`), with similarly poor results
  - Still working on this part, will probably just write my own loop instead of using PPO trainer to debug.

---

