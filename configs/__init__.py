# configs/__init__.py

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pretrained_models_path = {
    'bert-base-uncased': os.path.join(PROJECT_ROOT, 'models/bert-base-uncased'),
    'bert-large-uncased': os.path.join(PROJECT_ROOT, 'models/bert-large-uncased'),
}

video_feats_map = {
    'video_feats.pkl': 'swin-full',
    'swin_roi.pkl': 'swin-roi',
    'swin_feats.pkl': 'swin-full',
}

audio_feats_map = {
    'wavlm_feats.pkl': 'wavlm',
    'audio_feats.pkl': 'wav2vec2',
    'wavlm_feats-001.pkl': 'wavlm'
}

feat_dims = {
    'text': {
        'bert-base-uncased': 768,
        'bert-large-uncased': 1024
    },
    'video': {
        'resnet-50': 256,
        'swin-roi': 256,
        'swin-full': 1024,
        'raw_video': 768,
    },
    'audio': {
        'wavlm': 768,
        'wav2vec2': 768,
        'raw_audio': 768,
    }
}
