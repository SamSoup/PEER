python modelsv2/diagnostics.py \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset stsb \
  --cache_dir /scratch/06782/ysu707/ProtoLLM/stsb/Llama-3.1-8B-Instruct/cache \
  --ckpt /scratch/06782/ysu707/ProtoLLM/stsb/Llama-3.1-8B-Instruct/stageB.pt \
  --prototypes /scratch/06782/ysu707/ProtoLLM/stsb/Llama-3.1-8B-Instruct/prototypes.pt \
  --batch_size 64 \
  --max_length 256 \
  --save_json /scratch/06782/ysu707/ProtoLLM/stsb/Llama-3.1-8B-Instruct/diag.json
