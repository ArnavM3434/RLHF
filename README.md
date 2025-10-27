# RLHF Implementation (Based on InstructGPT)

I did this project to gain more familiarity with **Reinforcement Learning from Human Feedback (RLHF)** as a whole.  
It is based on the paper: [*Training language models to follow instructions with human feedback (InstructGPT)*](https://arxiv.org/abs/2203.02155).

---

## 🧩 Stages

### 1. Supervised Fine-Tuning (SFT)
- **Pretrained model:** `gpt2`  
- **LoRA configuration:** Model wrapped with a LoRA adapter (details in notebook)  
- **Dataset:** [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) — single-turn instruction–completion pairs in the form:

      Human: ...
      Assistant: ...

- **Training details:**
  - Loss computed on **completion only** (masked out prompts and padding tokens)
  - Training loss converged around **2.15** (started around ~2.7)
  - Results were consistent across multiple learning rates and LoRA configs  
- **Outcome:**
  - Much better **BLEU score** and **qualitative completions** than pretrained GPT-2  
  - See training runs and examples in the notebook

**Weights & Biases dashboard:**
<p align="center">
  <img src="https://i.imgur.com/Smm5Ql8.png" alt="Training loss curve" width="600"/>
</p>

---

### 2. Reward Model
- **Dataset:** `Dahoas/rm-static`  
- **Loss function:** Bradley–Terry  
- **Architecture:** Added a reward head to the SFT model  
- **Observations:**
  - Training was unstable — validation accuracy fluctuated significantly  
  - Likely due to data distribution mismatch (multi-turn human–assistant examples)

---

### 3. Proximal Policy Optimization (PPO)
- **Prompts:** Alpaca dataset  
- **Config:** See PPO configuration in notebooks  
- **Observations:**
  - KL divergence remained **negative**, despite various adjustments:
    - Reward normalization  
    - Generation tweaks  
    - Different KL coefficients  
  - Tried another reward model (`OpenAssistant/reward-model-deberta-v3-base`), with similarly poor results  

---

🚧 **Status:** Work in Progress
