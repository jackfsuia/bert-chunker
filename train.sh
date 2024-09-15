#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DEBUGPY=False
function usage(){
    echo 'Usage: bash finetune/finetune_ds.sh [-m MODEL_PATH] [-d DATA_PATH]'
}
while [[ "$1" != "" ]]; do
    case $1 in
        -b | --debug )
            shift
            DEBUGPY=True
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

# echo "The value of my_variable is: $GPUS_PER_NODE and $NNODES"
# torchrun $DISTRIBUTED_ARGS finetune_new.py \
python train_chunk_model.py \
  --base_model_name_or_path "nreimers/MiniLM-L6-H384-uncased" \
  --data_path "mixed_passenge_train_30000.json" \
  --eval_data_path "mixed_passenge_eval_30000.json" \
  --output_dir "outputModels" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 235 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "epoch" \
  --evaluate_before_train False \
  --save_strategy "epoch" \
  --save_steps 50 \
  --save_total_limit 20 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --log_level 'info'\
  --logging_dir "outputModels/logs" \
  --logging_strategy "epoch" \
  --report_to "tensorboard" \
  --model_max_length 255 \
  --lazy_preprocess True \
  --gradient_checkpointing False \
