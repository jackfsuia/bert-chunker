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
  --base_model_name_or_path "/data/all-MiniLM-L6-v2" \
  --data_path "/data/bert-chunker-v2/dataset/fw-train-cross.parquet" \
  --slide_window False\
  --eval_data_path1 "/data/bert-chunker-v2/no-dougao_no-oneline_test_10000.json" \
  --eval_data_path2 "/data/bert-chunker-v2/dataset/fw-test-cross.parquet" \
  --output_dir "/data/bert-chunker-v2/cross-again" \
  --num_train_epochs 10 \
  --per_device_train_batch_size 256 \
  --per_device_eval_batch_size 235 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "steps" \
  --eval_steps 50 \
  --evaluate_before_train True \
  --save_strategy "steps" \
  --save_steps 50 \
  --save_total_limit 20 \
  --learning_rate 1e-4 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --log_level 'info'\
  --logging_dir "/data/bert-chunker-v2/cross-again/logs" \
  --logging_strategy "steps" \
  --logging_step 50 \
  --report_to "wandb" \
  --model_max_length  255 \
  --gradient_checkpointing False \
  --metric_for_best_model "eval_fw-test-cross_loss"\
  --load_best_model_at_end True\
  --project_name 'bert_chunker'\
  --run_name 'fw-cross-again'\
  --debugpy $DEBUGPY\
  --model_checkpoint_bin ""
