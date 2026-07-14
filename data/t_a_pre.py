import pickle
import numpy as np
import os
import logging


def get_modal_data(args, data_args, modal_name, feats_path):
    """
    通用模态数据加载
    args: 配置对象（包含 seq_len 属性）
    data_args: 数据配置字典（包含索引）
    modal_name: 'audio' 或 'video'
    feats_path: 特征文件路径
    """
    logger = logging.getLogger(args.logger_name)

    if not os.path.exists(feats_path):
        raise Exception(f'Error: {modal_name} features directory is empty.')

    logger.info(f'Load {modal_name.capitalize()} Features Begin...')

    feats = load_feats(data_args, feats_path)

    # 根据模态获取对应的最大长度
    if modal_name == 'audio':
        max_seq_len = args.audio_seq_len
    elif modal_name == 'video':
        max_seq_len = args.video_seq_len
    else:
        raise ValueError(f"Unsupported modal: {modal_name}")

    data = padding_feats(feats, max_seq_len, modal_name)

    logger.info(f'Load {modal_name.capitalize()} Features Finished...')

    return data


def load_feats(data_args, feats_path):
    """加载特征文件"""
    with open(feats_path, 'rb') as f:
        feats = pickle.load(f)

    train_feats = [np.array(feats[x]) for x in data_args['train_data_index']]
    dev_feats = [np.array(feats[x]) for x in data_args['dev_data_index']]
    test_feats = [np.array(feats[x]) for x in data_args['test_data_index']]

    return {
        'train': train_feats,
        'dev': dev_feats,
        'test': test_feats
    }


def padding(feat, max_length, padding_mode='zero', padding_loc='end'):
    """对单个特征进行padding或截断"""
    assert padding_mode in ['zero', 'normal']
    assert padding_loc in ['start', 'end']

    length = feat.shape[0]

    if length > max_length:
        return feat[:max_length, :]

    if padding_mode == 'zero':
        pad = np.zeros([max_length - length, feat.shape[-1]])
    elif padding_mode == 'normal':
        mean, std = feat.mean(), feat.std()
        pad = np.random.normal(mean, std, (max_length - length, feat.shape[1]))

    if padding_loc == 'start':
        return np.concatenate((pad, feat), axis=0)
    else:
        return np.concatenate((feat, pad), axis=0)


def padding_feats(feats, max_seq_len, modal_name=''):
    """对所有特征进行padding"""
    p_feats = {}

    for dataset_type in feats.keys():
        f = feats[dataset_type]
        tmp_list = []
        length_list = []

        for i, x in enumerate(f):
            x_f = np.array(x)
            x_f = x_f.squeeze(1) if x_f.ndim == 3 else x_f

            # 调试信息
            if i == 0 and dataset_type == 'train':
                print(f"{modal_name} first train feat shape: {x_f.shape}")

            length_list.append(len(x_f))
            p_feat = padding(x_f, max_seq_len)
            tmp_list.append(p_feat)

        p_feats[dataset_type] = {
            'feats': tmp_list,
            'lengths': length_list
        }

    return p_feats


# ========== 便捷函数（保持原接口兼容）==========
def get_a_data(args, data_args):
    """加载音频数据"""
    return get_modal_data(args, data_args, 'audio', data_args['audio_feats_path'])


def get_v_data(args, data_args):
    """加载视频数据"""
    return get_modal_data(args, data_args, 'video', data_args['video_feats_path'])