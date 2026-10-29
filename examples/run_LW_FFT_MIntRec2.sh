#!/usr/bin bash

for seed in 0 1 2 3 4
do
    python run.py \
    --dataset 'MIntRec2.0' \
    --config_file_name 'LW_FFT_MIntRec2.py' \
    --video_data_path "" \
    --audio_data_path "" \
    --video_feats_path 'swin_roi.pkl' \
    --audio_feats_path 'wavlm_feats-001.pkl' \
    --text_backbone 'bert-large-uncased' \
    --method 'LW_FFT' \
    --train \
    --save_model \
    --save_results \
    --seed $seed \
    --num_workers 0 \
    --results_file_name 'results_lw_fft_mintrec2.csv'
done