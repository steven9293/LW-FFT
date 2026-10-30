import torch
from torch import nn
import torch.nn.functional as F



# 1. Wavelet domain contrast loss
class WaveletContrastLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(WaveletContrastLoss, self).__init__()
        self.temperature = temperature

    def forward(self, approx, detail, labels):
        # Add numerical stability
        approx_norm = F.normalize(approx, dim=1, eps=1e-8)
        detail_norm = F.normalize(detail, dim=1, eps=1e-8)
        features = approx_norm + detail_norm
        features = F.normalize(features, dim=1, eps=1e-8)

        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        sim_matrix = torch.clamp(sim_matrix, -10.0, 10.0)

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        eye_mask = torch.eye(mask.shape[0]).to(features.device)
        mask = mask - eye_mask

        exp_sim = torch.exp(sim_matrix) * (1 - eye_mask)
       
        log_prob = sim_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
        
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
        loss = -(mask * log_prob).sum(1) / mask_sum

        return loss.mean()



# 2. Wavelet domain sparsity loss
class WaveletSparsityLoss(nn.Module):
    def __init__(self):
        super(WaveletSparsityLoss, self).__init__()

    def forward(self, *coeffs):
        loss = 0.0
        n = 0
        for coeff in coeffs:
            if coeff is not None:
               
                l1_norm = torch.norm(coeff, p=1)
               
                loss += torch.clamp(l1_norm / (coeff.numel() + 1e-8), max=10.0)
                n += 1
        return loss / n if n > 0 else torch.tensor(0.0)



# 3. Time-frequency consistency loss
class TimeFreqConsistencyLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(TimeFreqConsistencyLoss, self).__init__()
        self.temperature = temperature

    def forward(self, time_feat, freq_feat, labels):
        time_norm = F.normalize(time_feat, dim=1, eps=1e-8)
        freq_norm = F.normalize(freq_feat, dim=1, eps=1e-8)

        sim_pos = (time_norm * freq_norm).sum(1) / self.temperature
        sim_pos = torch.clamp(sim_pos, -10.0, 10.0)

        sim_neg = torch.matmul(time_norm, freq_norm.T) / self.temperature
        sim_neg = torch.clamp(sim_neg, -10.0, 10.0)
        eye_mask = torch.eye(sim_neg.shape[0]).to(time_feat.device)
        sim_neg = sim_neg * (1 - eye_mask)

        loss = -torch.log(sim_pos.exp() / (sim_pos.exp() + sim_neg.exp().sum(1) + 1e-8) + 1e-8)
        return loss.mean()



# 4. Routing Mask Constraint Loss
class RoutingMaskLoss(nn.Module):
      def __init__(self):
        super().__init__()

    def forward(self, outputs):
        device = next(iter(outputs.values())).device if outputs else torch.device('cpu')
        loss = torch.tensor(0.0).to(device)
        count = 0

        # 1. Mask sparsity constraint: Encourages the mask mean to be close to 0.5
        if 'mask_mean' in outputs:
            mask_mean = outputs['mask_mean']
          
            loss_sparsity = (mask_mean - 0.5) ** 2
            loss = loss + loss_sparsity
            count += 1

        # 2. Mask variance constraint: encourages masks to be distinctive
        if 'mask_std' in outputs:
            mask_std = outputs['mask_std']
            
            loss_var = torch.exp(-mask_std) 
            loss = loss + loss_var
            count += 1

        # 3. Intent Entropy Constraint: Encourages clear intent selection 
        if 'intent_entropy' in outputs:
            intent_entropy = outputs['intent_entropy']
            
            loss_entropy = intent_entropy
            loss = loss + loss_entropy
            count += 1

        # 4. Constraint on characteristic energy conservation
        if 'z_energy' in outputs and 'z_masked_energy' in outputs:
            z_energy = outputs['z_energy']
            z_masked_energy = outputs['z_masked_energy']
            
            energy_ratio = z_masked_energy / (z_energy + 1e-8)
           
            # Target ratio is close to 0.5
            loss_energy = (energy_ratio - 0.5) ** 2
            loss = loss + loss_energy
            count += 1

        return loss / max(count, 1)



# 5. Intent Routing Regularization Loss
class IntentRoutingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels):
        device = labels.device
        loss = 0.0
        count = 0

        # 1. Intent Weight Entropy Regularization
        if 'intent_weights' in outputs:
            intent_weights = outputs['intent_weights']
           
            entropy = -(intent_weights * torch.log(intent_weights + 1e-8)).sum(-1).mean()
            loss += 0.01 * entropy
            count += 1

        # 2. Mask sparsity regularization
        if 'intent_mask' in outputs:
            mask = outputs['intent_mask']
            sparsity = mask.mean(dim=1).mean()
            loss += 0.01 * sparsity
            count += 1

        return loss / max(count, 1)



# 6. Total loss
class LWFFTLoss(nn.Module):
    def __init__(self, args=None):
        super(LWFFTLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

        temperature = getattr(args, 'temperature', 0.5) if args else 0.5
        args = args if args is not None else {}
        self.num_coarse = getattr(args, 'n_coarse_classes', 5) if args else 5

        # Loss weight
        self.w_contrast = getattr(args, 'w_contrast', 0.5) if args else 0.5
        self.w_sparsity = getattr(args, 'w_sparsity', 0.003) if args else 0.003
        self.w_tf = getattr(args, 'w_tf', 0.5) if args else 0.5
        self.w_routing = getattr(args, 'w_routing', 0.005) if args else 0.005        
        self.w_mask = getattr(args, 'w_mask', 0.5) if args else 0.5

        self.wavelet_contrast = WaveletContrastLoss(temperature=temperature)
        self.wavelet_sparsity = WaveletSparsityLoss()
        self.tf_consistency = TimeFreqConsistencyLoss(temperature=temperature)
        self.intent_routing = IntentRoutingLoss()       
        self.mask_loss = RoutingMaskLoss()

    def forward(self, outputs, labels):
        device = labels.device

        # 1. Main Category Loss 
        loss_main = self.ce(outputs['logits'], labels)

        # 2. Intent Assistance Loss
        loss_intent = torch.tensor(0.0).to(device)
        if 'intent_logits' in outputs:
            loss_intent = self.ce(outputs['intent_logits'], labels)

        # 3. Wavelet Coefficients 
        coeff_list = []
        for m in ['text', 'video', 'audio']:
            if f'approx_{m}' in outputs:
                coeff_list.append(outputs[f'approx_{m}'])
            if f'detail_{m}' in outputs:
                coeff_list.append(outputs[f'detail_{m}'])

        # 4. Wavelet Contrast Loss 
        loss_contrast = torch.tensor(0.0).to(device)
        if len(coeff_list) >= 2:
            try:
                approx = outputs.get('approx_text', None)
                detail = outputs.get('detail_text', None)
                if approx is not None and detail is not None:
                    loss_contrast = self.wavelet_contrast(approx.mean(dim=1), detail.mean(dim=1), labels)
            except:
                loss_contrast = torch.tensor(0.0).to(device)

        # 5. Sparsity Loss 
        loss_sparsity = self.wavelet_sparsity(*coeff_list)

        # 6. Time-frequency consistency loss
        loss_tf = torch.tensor(0.0).to(device)
        if 'fused_wavelet' in outputs and 'fft_global' in outputs:
            try:
                loss_tf = self.tf_consistency(
                    outputs['fused_wavelet'].mean(dim=1),
                    outputs['fft_global'],
                    labels
                )
            except:
                loss_tf = torch.tensor(0.0).to(device)

        # 7. Intent Routing Normalization 
        loss_routing = self.intent_routing(outputs, labels)

        # 8. Routing Mask Constraint Loss 
        loss_mask = self.mask_loss(outputs)

        # 9. Total Loss 
        total = (loss_main +
                 self.w_contrast * loss_contrast +
                 self.w_sparsity * loss_sparsity +
                 self.w_tf * loss_tf +
                 self.w_routing * loss_routing +
                 self.w_mask * loss_mask)

      

        return total
