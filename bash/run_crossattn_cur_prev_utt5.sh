#!/bin/bash
#SBATCH --job-name=check_txt
#SBATCH --qos=a100-4
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/dilab/Seong/peft-ser/out/prev_cur_5.out


modal='multimodal'
dataset=iemocap6
is_key_lora="True"

num_epochs=30
hidden_dim=256
speaker=wavlm
# split_data_dir=train_split_iemocap5


split_data_dir=train_split/train_split_prev_cur_5 #train_split_cp_prev5_sex
txt_len=256 # 256 (5 uttr) 128 (2 uttr)

cd ../experiment


cross_modal_atten="True"
exp_dir="iemocap_prev_cur_5_256_cls"
python finetune_emotion_multimodal_cross_modal_custom3.py --text_model roberta-large --audio_model whisper-medium \
    --dataset $dataset \
    --num_epochs $num_epochs --speaker $speaker --exp_dir $exp_dir --is_key_lora $is_key_lora \
    --learning_rate 0.0005 --modal $modal --cross_modal_atten $cross_modal_atten \
    --finetune_method 'lora' --finetune_roberta 'True' --split_data_dir $split_data_dir --max_txt_len $txt_len \
    --lora_rank 16 --lora_alpha 16 --lora_dropout 0.1 --lora_target_modules "key","query","value" \


exp_dir="iemocap_prev_cur_5_256_mean"
python finetune_emotion_multimodal_cross_modal_custom3_mean.py --text_model roberta-large --audio_model whisper-medium \
    --dataset $dataset \
    --num_epochs $num_epochs --speaker $speaker --exp_dir $exp_dir --is_key_lora $is_key_lora \
    --learning_rate 0.0005 --modal $modal --cross_modal_atten $cross_modal_atten \
    --finetune_method 'lora' --finetune_roberta 'True' --split_data_dir $split_data_dir --max_txt_len $txt_len \
    --lora_rank 16 --lora_alpha 16 --lora_dropout 0.1 --lora_target_modules "key","query","value" \


# # ffn + kqv 
# best acc
# cross_modal_atten="True"
# exp_dir="iemocap_prev_cur_5_256_cls"
# python finetune_emotion_multimodal_cross_modal_custom3.py --text_model roberta-large --audio_model whisper-medium \
#     --dataset $dataset \
#     --num_epochs $num_epochs --speaker $speaker --exp_dir $exp_dir --is_key_lora $is_key_lora \
#     --learning_rate 0.0005 --modal $modal --cross_modal_atten $cross_modal_atten \
#     --finetune_method 'lora' --finetune_roberta 'True' --split_data_dir $split_data_dir --max_txt_len $txt_len \
#     --lora_rank 16 --lora_alpha 16 --lora_dropout 0.1 --lora_target_modules "key","query","value"


# exp_dir="iemocap_prev_cur_5_256_mean"
# python finetune_emotion_multimodal_cross_modal_custom3_mean.py --text_model roberta-large --audio_model whisper-medium \
#     --dataset $dataset \
#     --num_epochs $num_epochs --speaker $speaker --exp_dir $exp_dir --is_key_lora $is_key_lora \
#     --learning_rate 0.0005 --modal $modal --cross_modal_atten $cross_modal_atten \
#     --finetune_method 'lora' --finetune_roberta 'True' --split_data_dir $split_data_dir --max_txt_len $txt_len \
#     --lora_rank 16 --lora_alpha 16 --lora_dropout 0.1 --lora_target_modules "key","query","value"
