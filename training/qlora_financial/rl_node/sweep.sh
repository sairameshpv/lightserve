# Learning-rate trials: for each "method rate", train from v3 (fresh adapter),
# convert the adapter, serve v3 + adapter, score greedy on val, log one row.
# Usage: bash sweep.sh "dpo 2e-5" "dpo 5e-5" ...   Results: ~/rl/sweep/results.jsonl
R=/home/ubuntu/rl; mkdir -p $R/sweep
MOUNTS="-v /root/.cache/huggingface:/root/.cache/huggingface -v $R:/rl \
 -v /home/ubuntu/openrlhf_src/openrlhf/trainer/ray/ppo_actor.py:/usr/local/lib/python3.12/dist-packages/openrlhf/trainer/ray/ppo_actor.py:ro \
 -v /home/ubuntu/openrlhf_src/openrlhf/cli/train_ppo_ray.py:/usr/local/lib/python3.12/dist-packages/openrlhf/cli/train_ppo_ray.py:ro"
RUN="sudo docker run --rm --gpus all --ipc=host --shm-size 16g -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/rl/code $MOUNTS -w /rl"
for trial in "$@"; do
  set -- $trial; method=$1; lr=$2; name=${method}_$lr; out=/rl/out/sweep/$name; log=$R/sweep/$name.log
  if [ $method = dpo ]; then
    sed -e "s#--eval.dataset /rl/data/dpo_val.jsonl --eval.split train ##; s#--eval.steps 21#--eval.steps -1#" \
        -e "s#--train.max_epochs 2#--train.max_epochs 1#; s#--adam.lr 5e-6#--adam.lr $lr#; s#/rl/out/dpo#$out#g" $R/rl_dpo.sh > $R/sweep/$name.sh
  else
    sed -e "s#--eval.dataset /rl/data/val.jsonl --eval.split train --eval.steps 16 --eval.temperature 0 --eval.n_samples_per_prompt 1 ##" \
        -e "s#/rl/data/train_1k.jsonl#/rl/data/trial_256.jsonl#; s#--actor.adam.lr 2e-5#--actor.adam.lr $lr#; s#/rl/out/grpo#$out#g" $R/rl_grpo.sh > $R/sweep/$name.sh
  fi
  t0=$(date +%s)
  $RUN --name sweep_train --entrypoint bash openrlhf:0.11.2 /rl/sweep/$name.sh > $log 2>&1
  t1=$(date +%s)
  $RUN --entrypoint python openrlhf:0.11.2 -c "
import torch, shutil
from safetensors.torch import save_file
sd = torch.load('$out/adapter_model.bin', map_location='cpu')
save_file({k: v.contiguous() for k, v in sd.items()}, '$out/adapter_model.safetensors')
" >> $log 2>&1
  sudo docker run -d --name sweep_serve --gpus all --ipc=host --network host -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/rl/code -v $R:/rl -w /rl \
    --entrypoint bash openrlhf:0.11.2 -c "python -m vllm.entrypoints.openai.api_server --model /rl/v3_merged --served-model-name v3 \
    --enable-lora --lora-modules v2=$out --max-lora-rank 16 --max-model-len 4096 --port 8000 --gpu-memory-utilization 0.9 > /rl/sweep/${name}_serve.log 2>&1" > /dev/null
  for i in $(seq 1 90); do curl -sf localhost:8000/v1/models > /dev/null && break; sleep 5; done
  stats=$(sudo docker exec -w /rl sweep_serve python /rl/val_greedy.py /rl/sweep/${name}_val.json /rl/out/val_v3.json 2>/dev/null | tail -1)
  sudo docker rm -f sweep_serve > /dev/null
  kl=$(grep -oE "Global step [0-9]+: \{[^}]*" $log | tail -1 | grep -oE "'kl': [-0-9.e]+" | head -1 | cut -d' ' -f2)
  dpo=$(tr '\r' '\n' < $log | grep -oE "loss=[^,]+, acc=[^,]+, chosen_reward=[^,]+, reject_reward=[^,]+" | tail -1)
  echo "{\"trial\": \"$name\", \"train_s\": $((t1 - t0)), \"kl\": \"$kl\", \"dpo_last\": \"$dpo\", \"val\": ${stats:-null}}" >> $R/sweep/results.jsonl
done
echo "$(date +%T) sweep done" >> $R/sweep/results.jsonl
