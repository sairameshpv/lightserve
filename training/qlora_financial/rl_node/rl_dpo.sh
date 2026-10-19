# DPO, quarter scale: 338 pairs from the 1k subset (4 answers/question), 2 passes, LoRA r16.
deepspeed --module openrlhf.cli.train_dpo \
   --model.model_name_or_path /rl/v3_merged \
   --data.dataset /rl/data/dpo_train_1k.jsonl --eval.dataset /rl/data/dpo_val.jsonl --eval.split train \
   --data.prompt_key prompt --data.chosen_key chosen --data.rejected_key rejected \
   --data.apply_chat_template --data.max_len 4096 \
   --model.beta 0.1 --adam.lr 5e-6 \
   --ds.lora.rank 16 --ds.lora.alpha 32 \
   --ds.zero_stage 3 --ref.offload --ds.param_dtype bf16 \
   --ds.packing_samples --ds.attn_implementation flash_attention_2 \
   --model.gradient_checkpointing_enable \
   --train.micro_batch_size 1 --train.batch_size 16 --train.max_epochs 2 \
   --logger.logging_steps 1 --eval.steps 21 --ckpt.save_steps -1 --ckpt.disable_ds \
   --ckpt.output_dir /rl/out/dpo --logger.tensorboard_dir /rl/out/dpo/tb
