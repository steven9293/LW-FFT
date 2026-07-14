class Param():

    def __init__(self, args):
        self.hyper_param = self._get_hyper_parameters(args)

    def _get_hyper_parameters(self, args):
        """
        LW-MIR Framework Hyperparameters for MIntRec Dataset

        MIntRec-specific optimizations:
        - Larger batch size for diverse intent samples
        - Higher learning rate for faster convergence
        - Stronger dropout for regularization
        - Enhanced frequency enhancement for subtle intent cues
        """
        hyper_parameters = {
            # ========== Common Parameters ==========
            'feats_processing_type': 'padding',
            'padding_mode': 'zero',
            'padding_loc': 'end',
            'need_aligned': True,
            'eval_monitor': ['f1'],
            'train_batch_size': 8,
            'eval_batch_size': 8,
            'test_batch_size': 8,
            'wait_patience': 8,
            'aligned_method': ['ctc'],
            'num_train_epochs': 100,
            'text_feats_dim': 1024,  # BERT large 输出维度
            'video_feats_dim': 1024,  # Swin Transformer 输出维度
            'audio_feats_dim': 768,  # WavLM 输出维度
            'weight_hidden_dim': 384,  # 隐藏层维度
            'weight_dropout': 0.2,
            'max_cons_seq_length': 30,

            # ========== Optimizer Parameters ==========
            'warmup_proportion': 0.1,
            'grad_clip': [1.0],
            'lr': [5e-6],
            'weight_decay': 0.1,


            # ========== Learnable Wavelet Parameters ==========
            'wavelet_kernel_types': ['morlet', 'mexican_hat', 'db4', 'sym5', 'coif4', 'bior4.4'],
            'n_wavelet_kernels': 6,
            'wavelet_scale_init': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
            'learnable_scale': True,
            'learnable_translation': True,
            'wavelet_decomposition_level': 4,  # Deeper decomposition for MIntRec
            'wavelet_freq_bands': [2, 4, 8, 16],
            'need_aligned': True,
            'aligned_method': 'ctc',
            # ========== WT-FFT Dual-Domain Fusion ==========
            'use_fft_branch': True,
            'fft_hidden_dim': 384,  # Larger for richer representations
            'wt_hidden_dim': 384,
            'cross_attn_heads': [12],
            'cross_attn_dropout': [0.15],
            'domain_fusion_method': 'gated',  # Gated fusion for better control

            # ========== Adaptive Frequency Enhancement ==========
            'use_freq_enhance': True,
            'highpass_cutoff': [0.25],  # Lower cutoff for more high-frequency details
            'freq_enhance_lambda_init': 0.6,
            'freq_enhance_learnable': True,
            'use_content_adaptive_lambda': True,

            # ========== Modality Credibility-Aware Fusion ==========
            'use_credibility_gate': True,
            'credibility_metrics': ['wavelet_energy', 'coefficient_entropy',
                                    'snr_estimate', 'spectral_flatness'],
            '''可调'''
            'credibility_temperature': [0.3],
            'low_credibility_threshold': 0.25,
            'dynamic_fusion_type': 'gated',  # Gated fusion for MIntRec

            # ========== Hierarchical Intent Decoding ==========
            'use_hierarchical_decoding': True,
            'n_coarse_classes': 5,  # MIntRec coarse: Persona, Query, Opinion, etc.
            'n_fine_classes': 20,  # Fine-grained intents
            'coarse_feature_ratio': 0.25,
            'fine_feature_ratio': 0.75,
            'hierarchical_loss_weight': [0.35, 0.65],



            # ========== Transformer Backbone Parameters ==========
            'use_ctx': True,
            'prompt_len': 5,  # MIntRec uses 'prompt_len' instead of 'ctx_len'
            'nheads': [12],
            'n_levels': [6],
            'attn_dropout': [0.15],
            'relu_dropout': 0.0,
            'embed_dropout': [0.3],
            'res_dropout': 0.15,
            'attn_mask': True,
            'ff_dim': 3072,

            # ========== SupCon Loss Parameters ==========
            'loss': 'SupCon',
            'temperature': 0.08,
            'base_temperature': 0.08,
            'contrastive_mode': 'hierarchical',  # Hierarchical contrastive learning

            # ========== Intent-Specific Parameters ==========
            'beta_shift': [0.01],
            'dropout_prob': [0.5],
            'shared_dim': [512],
            'eps': 1e-8,


            
        }
        return hyper_parameters