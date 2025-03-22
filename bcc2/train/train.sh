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
  --base_model_name_or_path "/data/bc-chinese-2/bge-small-zh-v1.5" \
  --data_path "/data/bc-chinese-2/newline_750k.parquet" \
  --slide_window True\
  --API_syn_dataset True\
  --eval_data_path "/data/bc-chinese-2/newline_10k.parquet" \
  --output_dir "/data/bc-chinese-2/newline_750k" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 70 \
  --per_device_eval_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "steps" \
  --eval_steps 1600 \
  --evaluate_before_train True \
  --save_strategy "steps" \
  --save_steps 1600 \
  --save_total_limit 2 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --log_level 'info'\
  --logging_dir "/data/bc-chinese-2/newline_750k/logs" \
  --logging_strategy "steps" \
  --logging_step 1600 \
  --report_to "wandb" \
  --model_max_length  512 \
  --gradient_checkpointing False \
  --metric_for_best_model "eval_newline_10k_loss"\
  --load_best_model_at_end True\
  --project_name 'bert_chunker'\
  --run_name 'newline_1000k'\
  --num_proc 10\
  --debugpy $DEBUGPY\
  --cache_dir "/data/bc-chinese-2/dataset/cache"\
  --model_checkpoint_bin ""
