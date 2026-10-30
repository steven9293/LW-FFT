import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft



class LearnableWaveletLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_kernels=6, kernel_size=3):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.alpha = nn.Parameter(torch.randn(num_kernels))
        self.scale = nn.Parameter(torch.randn(num_kernels))
        self.translation = nn.Parameter(torch.randn(num_kernels))
        self.proj_approx = nn.Linear(hidden_dim, hidden_dim)
        self.proj_detail = nn.Linear(hidden_dim, hidden_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
            for _ in range(num_kernels)
        ])

    def forward(self, x):
        x = x.float()
        x = self.input_proj(x)
        B, T, D = x.shape
        aggregated = 0
        x_perm = x.permute(0, 2, 1)
        for k in range(self.num_kernels):
            conv_out = self.convs[k](x_perm)
            aggregated = aggregated + self.alpha[k] * conv_out
        aggregated = aggregated.permute(0, 2, 1)
        approx = self.proj_approx(aggregated)
        detail = self.proj_detail(aggregated)
        return approx, detail



class TimeFrequencyEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.wavelet = LearnableWaveletLayer(input_dim, hidden_dim)
        self.tf_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.tf_norm = nn.LayerNorm(hidden_dim)
        self.seq_linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.seq_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, dropout=dropout),
            num_layers=2
        )
        self.align_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        approx, detail = self.wavelet(x)
        tf_out, _ = self.tf_attn(approx, detail, detail)
        tf_out = self.tf_norm(tf_out + approx)
        seq_feat = self.seq_linear(x)
        seq_out = self.seq_transformer(seq_feat)
        combined = torch.cat([tf_out, seq_out], dim=-1)
        aligned = self.align_proj(combined)
        fused, _ = self.cross_attn(aligned, aligned, aligned)
        fused = self.cross_norm(fused + aligned)
        return fused, approx, detail



class AdaptiveFrequencyEnhancement(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        self.hpf = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        nn.init.constant_(self.hpf.weight, 0)
        self.hpf.weight.data[:, :, 1] = 1.0
        self.hpf.weight.data[:, :, 0] = -0.5
        self.hpf.weight.data[:, :, 2] = -0.5

    def forward(self, x):
        B, T, H = x.shape
        x_perm = x.permute(0, 2, 1)
        high_freq = self.hpf(x_perm).permute(0, 2, 1)
        lambda_coef = self.gate(x.mean(dim=1, keepdim=True))
        enhanced = x + lambda_coef * high_freq
        return enhanced


class IntentAwareDimensionMask(nn.Module):
    def __init__(self, hidden_dim=384, num_intents=20, temperature=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_intents = num_intents
        self.temperature = temperature

     
        self.intent_prototypes = nn.Parameter(torch.randn(num_intents, hidden_dim) * 0.01)

      
        self.intent_matcher = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_intents)
        )

       
        self.dimension_mask_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )

       
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, z, return_all=False):
        B, H = z.shape
        temp = max(self.temperature, 0.05)

      
        intent_scores = self.intent_matcher(z)
        intent_weights = F.softmax(intent_scores / temp, dim=-1)
        intent_mask = torch.sigmoid(torch.matmul(intent_weights, self.intent_prototypes))

     
        dim_mask = self.dimension_mask_predictor(z)

        
        alpha = torch.sigmoid(self.alpha)
        final_mask = alpha * intent_mask + (1 - alpha) * dim_mask
        final_mask = torch.clamp(final_mask, 0.1, 0.9)

        z_masked = z * final_mask

        if return_all:
            return {
                'z_masked': z_masked,
                'final_mask': final_mask,
                'intent_mask': intent_mask,
                'dim_mask': dim_mask,
                'intent_weights': intent_weights,
               
                'mask_mean': final_mask.mean(),           
                'mask_std': final_mask.std(),             
                'intent_entropy': -(intent_weights * torch.log(intent_weights + 1e-8)).sum(-1).mean(),  
                'z_energy': z.norm(p=2, dim=-1).mean(),  
                'z_masked_energy': z_masked.norm(p=2, dim=-1).mean(),  
            }

        return z_masked, final_mask, intent_weights
        
class WaveletReliabilityFusion(nn.Module):
    def __init__(self, hidden_dim, num_modalities=3):
        super().__init__()
        self.num_modalities = num_modalities
        self.reliability_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def compute_entropy(self, feat):
        var = torch.var(feat, dim=1, keepdim=True)
        entropy = -torch.sum(var * torch.log(var + 1e-8), dim=-1, keepdim=True)
        return entropy

    def forward(self, features_list):
        weights = []
        for approx, detail in features_list:
            approx_pooled = approx.mean(dim=1)  # [B, H]
            detail_pooled = detail.mean(dim=1)  # [B, H]
            reliability_input = torch.cat([approx_pooled, detail_pooled], dim=-1)  # [B, 2H]
            reliability = self.reliability_net(reliability_input)
            weights.append(reliability)

        weights = torch.stack(weights, dim=1)
        weights = F.softmax(weights, dim=1)
        return weights



class LW_FFT(nn.Module):
    def __init__(self, args):
        super(LW_FFT, self).__init__()

        
        self.text_feats_dim = args.text_feats_dim
        self.video_feats_dim = args.video_feats_dim
        self.audio_feats_dim = args.audio_feats_dim
        self.hidden_dim = args.weight_hidden_dim
        self.dropout = args.weight_dropout
        self.temperature = args.temperature
        self.need_aligned = args.need_aligned
        self.num_labels = args.num_labels
        self.aligned_method = args.aligned_method
        self.num_coarse = getattr(args, 'n_coarse_classes', 5)

        if self.need_aligned:
            from .AlignNets import AlignSubNet
            self.alignNet = AlignSubNet(args, self.aligned_method)
        from .SubNets.FeatureNets import BERTEncoder
        self.text_backbone = BERTEncoder(args)

        
        self.text_encoder = TimeFrequencyEncoder(self.text_feats_dim, self.hidden_dim, self.dropout)
        self.video_encoder = TimeFrequencyEncoder(self.video_feats_dim, self.hidden_dim, self.dropout)
        self.audio_encoder = TimeFrequencyEncoder(self.audio_feats_dim, self.hidden_dim, self.dropout)

        
        self.fft_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

       
        self.cross_modal_attn = nn.MultiheadAttention(
            self.hidden_dim, num_heads=4, dropout=self.dropout, batch_first=True
        )

       
        self.afe_text = AdaptiveFrequencyEnhancement(self.hidden_dim)
        self.afe_video = AdaptiveFrequencyEnhancement(self.hidden_dim)
        self.afe_audio = AdaptiveFrequencyEnhancement(self.hidden_dim)

        
        self.reliability_fusion = WaveletReliabilityFusion(self.hidden_dim)

       
        self.intent_routing = IntentAwareDimensionMask(
            hidden_dim=self.hidden_dim,
            num_intents=self.num_labels,

            temperature=0.1
        )

        self.final_classifier = nn.Linear(self.hidden_dim, self.num_labels)
        

        self.fusion_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.num_labels),
            nn.Sigmoid()
        )

        
        self.temporal_attn = nn.Linear(self.hidden_dim, 1)

       
        self.lambda_coarse = getattr(args, 'lambda_coarse', 0.5)
        self.lambda_fine = getattr(args, 'lambda_fine', 0.5)
        self.text_adapter = nn.Linear(self.text_feats_dim, self.hidden_dim)
        self.video_adapter = nn.Linear(self.video_feats_dim, self.hidden_dim)
        self.audio_adapter = nn.Linear(self.audio_feats_dim, self.hidden_dim)



    def forward(self, text_feats, video_feats, audio_feats):
        text_feats = self.text_backbone(text_feats)
        audio_feats = audio_feats.float()
        video_feats = video_feats.float()

        if self.need_aligned:
            text_feat, audio_feat, video_feat = self.alignNet(text_feats, audio_feats, video_feats)
        else:
            text_feat, audio_feat, video_feat = text_feats, audio_feats, video_feats

       
        t_feat, t_approx, t_detail = self.text_encoder(text_feat)
        v_feat, v_approx, v_detail = self.video_encoder(video_feat)
        a_feat, a_approx, a_detail = self.audio_encoder(audio_feat)

       
        t_enh = self.afe_text(t_feat)
        v_enh = self.afe_video(v_feat)
        a_enh = self.afe_audio(a_feat)

       
        mod_weights = self.reliability_fusion([(t_approx, t_detail), (v_approx, v_detail), (a_approx, a_detail)])
        fused_wavelet = mod_weights[:, 0:1, :] * t_enh + mod_weights[:, 1:2, :] * v_enh + mod_weights[:, 2:3, :] * a_enh

        text_proj = self.text_adapter(text_feat.float())  
        video_proj = self.video_adapter(video_feat.float())  
        audio_proj = self.audio_adapter(audio_feat.float())  

       
        def compute_fft_feat(x):
            x_perm = x.permute(0, 2, 1)  # [B, H, T]
            fft_out = torch.fft.rfft(x_perm, dim=-1).real 
            fft_proj = self.fft_proj(fft_out.mean(dim=-1)) 
            return fft_proj

        t_fft = compute_fft_feat(text_proj)
        v_fft = compute_fft_feat(video_proj)
        a_fft = compute_fft_feat(audio_proj)
        fft_global = (t_fft + v_fft + a_fft) / 3.0  

       
        fft_global_expanded = fft_global.unsqueeze(1).expand(-1, fused_wavelet.shape[1], -1)
        fused_cross, _ = self.cross_modal_attn(
            query=fft_global_expanded,
            key=fft_global_expanded,
            value=fused_wavelet
        )
        fused_cross_pooled = fused_cross.mean(dim=1) 

        
        routing_outputs = self.intent_routing(fused_cross_pooled, return_all=True)
        z_masked = routing_outputs['z_masked']
        final_mask = routing_outputs['final_mask']
        intent_weights = routing_outputs['intent_weights']


        final_logits =self.final_classifier(z_masked)

       
        t_attn = torch.sigmoid(self.temporal_attn(t_feat))
        v_attn = torch.sigmoid(self.temporal_attn(v_feat))
        a_attn = torch.sigmoid(self.temporal_attn(a_feat))
        temporal_weights = {
            'text': t_attn.mean(dim=1),
            'video': v_attn.mean(dim=1),
            'audio': a_attn.mean(dim=1)
        }

        outputs = {
            "logits": final_logits,

            "fused_wavelet": fused_wavelet,
            "modality_weights": mod_weights,
            "temporal_weights": temporal_weights,
            "fft_global": fft_global,
            "approx_text": t_approx,
            "detail_text": t_detail,
            "approx_video": v_approx,
            "detail_video": v_detail,
            "approx_audio": a_approx,
            "detail_audio": a_detail,
            "intent_mask": final_mask,
            "intent_weights": intent_weights,
            "mask_mean": routing_outputs['mask_mean'],
            "mask_std": routing_outputs['mask_std'],
            "intent_entropy": routing_outputs['intent_entropy'],
            "z_energy": routing_outputs['z_energy'],
            "z_masked_energy": routing_outputs['z_masked_energy'],


        }

        return outputs
