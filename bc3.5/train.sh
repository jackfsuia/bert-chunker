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
  --base_model_name_or_path "/hy-tmp/all-MiniLM-L6-v2" \
  --data_path "/hy-tmp/训练代码/doubao_35000_train_edg100.parquet" \
  --slide_window False\
  --API_syn_dataset True\
  --eval_data_path "/hy-tmp/训练代码/doubao_35000_eval_edg100.parquet" \
  --output_dir "/hy-tmp/checkpoints_edg100" \
  --num_train_epochs 8 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 70 \
  --gradient_accumulation_steps 8 \
  --eval_strategy "steps" \
  --eval_steps 50 \
  --evaluate_before_train True \
  --save_strategy "steps" \
  --save_steps 50 \
  --save_total_limit 2 \
  --learning_rate 1e-4 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --log_level 'info'\
  --logging_dir "/hy-tmp/checkpoints/logs" \
  --logging_strategy "steps" \
  --logging_step 10 \
  --model_max_length  255 \
  --gradient_checkpointing False\
  --report_to "wandb"\
  --metric_for_best_model "eval_doubao_35000_eval_edg100_loss"\
  --load_best_model_at_end True\
  --project_name 'bert_chunker'\
  --run_name 'doubao_35000_1228_edge100'\