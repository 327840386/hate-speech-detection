#!/bin/bash
# export PYTHONPATH=/root/whisper:$PYTHONPATH
export PYTHONPATH=/home/liu.ten/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

run_dir=/home/liu.ten/demo/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech

speech_encoder_path=/home/liu.ten/demo/SLAM-LLM/src/slam_llm/models/WavLM-Large.pt
llm_path=TinyLlama/TinyLlama-1.1B-Chat-v0.1
train_data_path=/home/liu.ten/demo/SLAM-LLM/examples/asr_librispeech/train_data.jsonl
val_data_path=/home/liu.ten/demo/SLAM-LLM/examples/asr_librispeech/validation_data.jsonl

# choose from exp1, exp2 and exp3
experiment_name=exp1

output_dir=/home/liu.ten/demo/tmp/${experiment_name}-TinyLlama-1.1B-librispeech-linear-steplrwarmupkeep1e-4-wavlm-large-$(date +"%Y%m%d")

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=TinyLlama-1.1B \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=2048 \
++model_config.encoder_name=wavlm \
++model_config.normalize=true \
++dataset_config.normalize=true \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1024 \
++model_config.encoder_projector=linear \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.use_data_augmentation=true \
++dataset_config.input_type=raw \
++log_config.use_wandb=true \
++log_config.wandb_exp_name=$experiment_name \
++train_config.model_name=hate_speech_detection \
++train_config.use_data_augmentation=true \
++train_config.experiment_type=$experiment_name \
++train_config.num_epochs=200 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=100 \
++train_config.batch_size_training=8 \
++train_config.val_batch_size=5 \
++train_config.num_workers_dataloader=2 \
++train_config.output_dir=$output_dir \
++metric=acc \
"

# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 1 \
        --master_port=29503 \
        $code_dir/finetune_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++train_config.use_fp16=true \
        $hydra_args
fi