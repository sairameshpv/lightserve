# Node scripts for the RL comparison

Copies of the scripts that produced the RL numbers in `../README.md`. They run on the
GPU node, not the Mac: the `.sh` files run inside the `openrlhf:0.11.2` container, with
`~/rl` mounted at `/rl` and this repo at `/rl/code`, and with `../openrlhf_patches.patch`
applied to the installed OpenRLHF. Paths such as `/rl/v3_merged` and `/rl/data/` are on the node.

- `merge_v3.py`: merges the v3 adapter into bf16 weights (`~/rl/v3_merged`) and checks
  that 3 greedy answers are unchanged.
- `rl_dpo.sh`, `rl_grpo.sh`: the quarter-scale DPO and GRPO runs (1,000 questions).
- `ppo_tiny.sh`: the 2-GPU PPO memory probe (critic on its own GPU).
- `sweep.sh`: learning-rate trials. For each trial it trains, serves v3 + adapter with vLLM,
  scores with `val_greedy.py` and appends a row to `~/rl/sweep/results.jsonl`.
- `val_greedy.py`: greedy accuracy on the 299 validation questions, plus answers changed,
  became right / became wrong and a sign-test p against v3.
