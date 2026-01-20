#!/bin/bash

base_score_path=./selected_frames/longvideobench/blip
score_type=selected_frames
dataset_name=longvideobench

python ./evaluation/change_score.py \
    --base_score_path $base_score_path \
    --score_type $score_type \
    --dataset_name $dataset_name 


export PYTHONPATH=./evaluation:/home/liuheming/lmms-eval:$PYTHONPATH

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_processes 3 --main_process_port 12345 evaluation/run_aks_eval.py \
    --model aks_qwen2_5_vl \
    --model_args pretrained=./checkpoints/Qwen2.5-VL-7B-Instruct,use_topk=True,nframes=64,use_flash_attention_2=False \
    --tasks longvideobench_val_v \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen2_5_vl_7b_lvb_v \
    --output_path ./results/${score_type}