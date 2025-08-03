#!bin/bash

set -x

PAPER_TABLE=(coco2017_cap_val flickr30k_test gqa \
mmbench_en_dev mme mmmu_val \
nocaps_val ok_vqa_val2014 pope \
scienceqa_img seedbench)
# max_pixels=12845056

PAPER_TABLE=(mme mmmu_val nocaps_val)

LOG_DIR=./workdir/lmmseval
RUN_NAME=orig_qwen_2.5_vl_7b

for TASK in ${PAPER_TABLE[@]}; do
    echo "------------Task $TASK is running------------"
    CUDA_VISIBLE_DEVICES=0,1,2 NLTK_DATA="/home/yrc/nltk_data" \
    accelerate launch \
        --num_processes=1 \
        -m lmms_eval \
        --model qwen2_5_vl \
        --model_args pretrained="./checkpoints/Qwen2.5-VL-7B-Instruct,max_pixels=147456,interleave_visuals=False,device_map=auto" \
        --tasks $TASK \
        --verbosity WARNING \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $RUN_NAME \
        --output_path $LOG_DIR/$RUN_NAME

    sleep 5
done