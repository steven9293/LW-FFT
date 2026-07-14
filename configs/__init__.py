# configs/__init__.py

import os

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 预训练模型路径
pretrained_models_path = {
    'bert-base-uncased': os.path.join(PROJECT_ROOT, 'models/bert-base-uncased'),
    'bert-large-uncased': os.path.join(PROJECT_ROOT, 'models/bert-large-uncased'),
}

# 视频特征映射
video_feats_map = {
    'video_feats.pkl': 'swin-full',
    'swin_roi.pkl': 'swin-roi',
    'swin_feats.pkl': 'swin-full',
}

# 音频特征映射
audio_feats_map = {
    'wavlm_feats.pkl': 'wavlm',
    'audio_feats.pkl': 'wav2vec2',
    'wavlm_feats-001.pkl': 'wavlm'
}

# 特征维度映射
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