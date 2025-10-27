I did this project to gain more familiarity with RLHF as a whole. It is based on the InstructGPT paper (https://arxiv.org/abs/2203.02155).

Stages

SFT
Pretrained model: "gpt2"
Wrapped in a LoRA Config (see more details in the notebook)
Fine tuned on Alpaca dataset (see more details in notebook) -> single turn instruction-completion prompts in the form "Human:... Assistant..."
Calculated loss on completion only -> masked out the prompts and padding tokens
Training loss converged around 2.15 (started around 2.7)
Training runs below

Reward Model
Used the 

PPO (Proximal Policy Optimization)



