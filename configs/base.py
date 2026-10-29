# configs/base.py
import importlib
from easydict import EasyDict
from .__init__ import pretrained_models_path, video_feats_map, audio_feats_map, feat_dims


class ParamManager:

    def __init__(self, args):
        # 先将 args 转换为字典
        if hasattr(args, 'keys'):
            args_dict = args.copy()
        else:
            args_dict = vars(args)

        # 设置特征维度和模型路径
        text_backbone = args_dict.get('text_backbone', 'bert-large-uncased')
        video_feats_path = args_dict.get('video_feats_path', 'video_feats.pkl')
        audio_feats_path = args_dict.get('audio_feats_path', 'audio_feats.pkl')

        # 设置 BERT 模型路径
        args_dict['text_backbone_path'] = pretrained_models_path[text_backbone]

        # 设置各模态特征维度
        args_dict['text_feats_dim'] = feat_dims['text'][text_backbone]
        args_dict['video_feats_dim'] = feat_dims['video'][video_feats_map[video_feats_path]]
        args_dict['audio_feats_dim'] = feat_dims['audio'][audio_feats_map[audio_feats_path]]

        self.args = EasyDict(args_dict)


def add_config_param(old_args, config_file_name=None):
    if config_file_name is None:
        config_file_name = old_args.config_file_name

    if config_file_name.endswith('.py'):
        module_name = '.' + config_file_name[:-3]
    else:
        module_name = '.' + config_file_name

    config = importlib.import_module(module_name, 'configs')

    config_param = config.Param
    method_args = config_param(old_args)

    # 将 old_args 转换为字典
    if hasattr(old_args, 'keys'):
        old_dict = old_args.copy()
    else:
        old_dict = vars(old_args)

    new_args = EasyDict(old_dict, **method_args.hyper_param)

    # 把长度为1的列表转成标量
    for key, value in list(new_args.items()):
        if isinstance(value, list) and len(value) == 1:
            new_args[key] = value[0]

    return new_args