class Param():
    def __init__(self, args):
        self.hyper_param = self._get_hyper_parameters(args)

    def _get_hyper_parameters(self, args):
        hyper_parameters = {
            # Common Parameters 
            'padding_mode': 'zero',
            'padding_loc': 'end',
            'need_aligned': True,
            'aligned_method': ['ctc'],
            'eval_monitor': ['f1'],

            # Training Parameters
            'train_batch_size': 8,
            'eval_batch_size': 8,
            'test_batch_size': 8,
            'wait_patience': 8,
            'num_train_epochs': 100,
            'warmup_proportion': 0.1,
            'grad_clip': [1.0],
            'lr': [5e-6],
            'weight_decay': 0.1,

            # Model Dimensions 
            'text_feats_dim': 1024,
            'video_feats_dim': 256,   
            'audio_feats_dim': 768,
            'weight_hidden_dim': 384,
            'weight_dropout': 0.2,
            'n_coarse_classes': 6,   

            # Loss & Align Parameters 
            'temperature': 0.08,
            'eps': 1e-8,

            # Loss Weights 
            'w_contrast': 0.5,
            'w_sparsity': 0.0001,
            'w_tf': 0.005,
            'w_routing': 0.005,
            'w_mask': 0.5,
        }
        return hyper_parameters
