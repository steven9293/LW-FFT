#!/usr/bin bash

for seed in 0 1 2 3 4
do
    python run.py \
    --dataset 'MIntRec' \
    --config_file_name 'LW_FFT_MIntRec.py' \
    --video_feats_path 'video_feats.pkl' \
    --audio_feats_path 'audio_feats.pkl' \
    --text_backbone 'bert-large-uncased' \
    --method 'LW_FFT' \
    --train \
    --save_model \
    --save_results \
    --seed $seed \
    --num_workers 0 \
    --results_file_name 'results_lw_fft_mintrec.csv'
done