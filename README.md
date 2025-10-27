I did this project to gain more familiarity with RLHF as a whole. It is based on the InstructGPT paper (https://arxiv.org/abs/2203.02155).

Stages

SFT
Pretrained model: gpt2
Wrapped in a LoRA Config (see more details in the notebook)
Fine tuned on Alpaca dataset (see more details in notebook) -> single turn instruction-completion prompts in the form "Human:... Assistant..."
Calculated loss on completion only -> masked out the prompts and padding tokens
Training loss converged around 2.15 (started around 2.7) -> similar results when trying multiple learning rates, adjusting LoRA configs, etc.
Training runs below
Achieved much better bleu score as well as much better completions compared to pretrained gpt2 (obviously)

Reward Model
Used the Dahoas/rm-static dataset
Used Bradley-Terry loss
Added a reward head to my SFT model
Unstable with validation accuracy fluctuating a lot (see notebook)
Likely related to the fact that the distribution for this data was different (multi turn human-assistant examples)

PPO (Proximal Policy Optimization)
Used the Alpaca dataset for prompts
See PPO Config in Notebooks
Didn't really work, KL divergence remained negative despite trying multiple things, including normalizing rewards, tweaking generations, tightening KL coefficients, etc.
Ended up trying a different reward model, which also did not give great results: OpenAssistant/reward-model-deberta-v3-base
THIS IS A WORK IN PROGRESS

