
set -x
PAPER_TABLE=coco2017_cap_val

LOG_DIR=./workdir/lmmseval

RUN_NAME=divprune_qwen_2.5_vl_7b_time
CUDA_VISIBLE_DEVICES=0,1,2 SUBSET_RATIO=0.098 EVAL_TIME=TRUE NLTK_DATA="/home/yrc/nltk_data" \
    accelerate launch \
        --num_processes=1 \
        -m lmms_eval \
        --model qwen2_5_vl \
        --model_args pretrained="./checkpoints/Qwen2.5-VL-7B-Instruct,device_map=auto,max_pixels=12845056,interleave_visuals=False" \
        --tasks $PAPER_TABLE \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $RUN_NAME \
        --output_path $LOG_DIR/$RUN_NAME | tee ./logs/lmmeval_qwen2.5_vl_divprune_time.log

sleep 10

RUN_NAME=orig_qwen_2.5_vl_7b_time
CUDA_VISIBLE_DEVICES=0,1,2 EVAL_TIME=TRUE NLTK_DATA="/home/yrc/nltk_data" \
    accelerate launch \
        --num_processes=1 \
        -m lmms_eval \
        --model qwen2_5_vl \
        --model_args pretrained="./checkpoints/Qwen2.5-VL-7B-Instruct,device_map=auto,max_pixels=12845056,interleave_visuals=False" \
        --tasks $PAPER_TABLE \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $RUN_NAME \
        --output_path $LOG_DIR/$RUN_NAME | tee ./logs/lmmeval_qwen2.5_vl_origin_time.log

echo "Divprune memory/latency"
python ./src/extract_time.py --path ./logs/lmmeval_qwen2.5_vl_divprune_time.log

echo "Original model memory/latency"
python ./src/extract_time.py --path ./logs/lmmeval_qwen2.5_vl_origin_time.log

