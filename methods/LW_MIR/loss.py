import torch
from torch import nn
import torch.nn.functional as F


# ============================================================
# 1. 小波域对比损失（添加数值稳定性）
# ============================================================
class WaveletContrastLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(WaveletContrastLoss, self).__init__()
        self.temperature = temperature

    def forward(self, approx, detail, labels):
        # 添加数值稳定性
        approx_norm = F.normalize(approx, dim=1, eps=1e-8)
        detail_norm = F.normalize(detail, dim=1, eps=1e-8)
        features = approx_norm + detail_norm
        features = F.normalize(features, dim=1, eps=1e-8)

        sim_matrix = torch.matmul(features, features.T) / self.temperature
        # 限制相似度范围防止溢出
        sim_matrix = torch.clamp(sim_matrix, -10.0, 10.0)

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        eye_mask = torch.eye(mask.shape[0]).to(features.device)
        mask = mask - eye_mask

        exp_sim = torch.exp(sim_matrix) * (1 - eye_mask)
        # 添加小值防止除零
        log_prob = sim_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)

        # 避免除以零
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
        loss = -(mask * log_prob).sum(1) / mask_sum

        return loss.mean()


# ============================================================
# 2. 小波域稀疏性损失（修复）
# ============================================================
class WaveletSparsityLoss(nn.Module):
    def __init__(self):
        super(WaveletSparsityLoss, self).__init__()

    def forward(self, *coeffs):
        loss = 0.0
        n = 0
        for coeff in coeffs:
            if coeff is not None:
                # 使用平均L1范数，防止数值过大
                l1_norm = torch.norm(coeff, p=1)
                # 限制最大值防止爆炸
                loss += torch.clamp(l1_norm / (coeff.numel() + 1e-8), max=10.0)
                n += 1
        return loss / n if n > 0 else torch.tensor(0.0)


# ============================================================
# 3. 时频一致性损失（添加数值稳定性）
# ============================================================
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


# ============================================================
# 4. 路由掩码约束损失（新增：防止掩码崩溃）
# ============================================================
class RoutingMaskLoss(nn.Module):
    """
    路由掩码约束损失
    1. 掩码稀疏性：鼓励掩码均值接近0.5
    2. 掩码方差：鼓励掩码有区分度
    3. 意图熵：鼓励意图选择清晰
    4. 特征能量保持：防止特征消失
    """
    def __init__(self):
        super().__init__()

    def forward(self, outputs):
        device = next(iter(outputs.values())).device if outputs else torch.device('cpu')
        loss = torch.tensor(0.0).to(device)
        count = 0

        # 1. 掩码稀疏性约束：鼓励掩码均值接近0.5
        if 'mask_mean' in outputs:
            mask_mean = outputs['mask_mean']
            # 目标：掩码均值接近0.5（平衡保留和抑制）
            loss_sparsity = (mask_mean - 0.5) ** 2
            loss = loss + loss_sparsity
            count += 1

        # 2. 掩码方差约束：鼓励掩码有区分度
        if 'mask_std' in outputs:
            mask_std = outputs['mask_std']
            # 目标：掩码标准差不要太小（否则所有维度都一样）
            loss_var = torch.exp(-mask_std)  # std越小，loss越大
            loss = loss + loss_var
            count += 1

        # 3. 意图熵约束：鼓励意图选择清晰（低熵）
        if 'intent_entropy' in outputs:
            intent_entropy = outputs['intent_entropy']
            # 目标：熵不要太大（不要均匀分布）
            loss_entropy = intent_entropy
            loss = loss + loss_entropy
            count += 1

        # 4. 特征能量保持约束：掩码后特征不要消失
        if 'z_energy' in outputs and 'z_masked_energy' in outputs:
            z_energy = outputs['z_energy']
            z_masked_energy = outputs['z_masked_energy']
            # 目标：掩码后能量保持在原始能量的30%~70%
            energy_ratio = z_masked_energy / (z_energy + 1e-8)
            # 目标比值接近0.5
            loss_energy = (energy_ratio - 0.5) ** 2
            loss = loss + loss_energy
            count += 1

        return loss / max(count, 1)


# ============================================================
# 5. 意图路由正则化损失（保留原有）
# ============================================================
class IntentRoutingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels):
        device = labels.device
        loss = 0.0
        count = 0

        # 1. 意图权重熵正则化
        if 'intent_weights' in outputs:
            intent_weights = outputs['intent_weights']
            # 添加小值防止log(0)
            entropy = -(intent_weights * torch.log(intent_weights + 1e-8)).sum(-1).mean()
            loss += 0.01 * entropy
            count += 1

        # 2. 掩码稀疏性正则化
        if 'intent_mask' in outputs:
            mask = outputs['intent_mask']
            sparsity = mask.mean(dim=1).mean()
            loss += 0.01 * sparsity
            count += 1

        return loss / max(count, 1)


# ============================================================
# 6. 总损失（完整版）
# ============================================================
class LWMIRLoss(nn.Module):
    def __init__(self, args=None):
        super(LWMIRLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

        temperature = getattr(args, 'temperature', 0.5) if args else 0.5
        self.num_coarse = getattr(args, 'n_coarse_classes', 5) if args else 5

        # 损失权重
        self.w_contrast = getattr(args, 'w_contrast', 0.5) if args else 0.5
        self.w_sparsity = getattr(args, 'w_sparsity', 0.0001) if args else 0.0001
        self.w_tf = getattr(args, 'w_tf', 0.005) if args else 0.005
        self.w_routing = getattr(args, 'w_routing', 0.005) if args else 0.005
        self.w_intent = getattr(args, 'w_intent', 0.0) if args else 0.0
        # ✅ 路由掩码约束权重
        self.w_mask = getattr(args, 'w_mask', 0.5) if args else 0.5

        self.wavelet_contrast = WaveletContrastLoss(temperature=temperature)
        self.wavelet_sparsity = WaveletSparsityLoss()
        self.tf_consistency = TimeFreqConsistencyLoss(temperature=temperature)
        self.intent_routing = IntentRoutingLoss()
        # ✅ 新增路由掩码约束
        self.mask_loss = RoutingMaskLoss()

    def forward(self, outputs, labels):
        device = labels.device

        # ========== 1. 主分类损失 ==========
        loss_main = self.ce(outputs['logits'], labels)

        # ========== 2. 意图辅助损失（如果有） ==========
        loss_intent = torch.tensor(0.0).to(device)
        if 'intent_logits' in outputs:
            loss_intent = self.ce(outputs['intent_logits'], labels)

        # ========== 3. 收集小波系数 ==========
        coeff_list = []
        for m in ['text', 'video', 'audio']:
            if f'approx_{m}' in outputs:
                coeff_list.append(outputs[f'approx_{m}'])
            if f'detail_{m}' in outputs:
                coeff_list.append(outputs[f'detail_{m}'])

        # ========== 4. 小波对比损失 ==========
        loss_contrast = torch.tensor(0.0).to(device)
        if len(coeff_list) >= 2:
            try:
                approx = outputs.get('approx_text', None)
                detail = outputs.get('detail_text', None)
                if approx is not None and detail is not None:
                    loss_contrast = self.wavelet_contrast(approx.mean(dim=1), detail.mean(dim=1), labels)
            except:
                loss_contrast = torch.tensor(0.0).to(device)

        # ========== 5. 稀疏性损失 ==========
        loss_sparsity = self.wavelet_sparsity(*coeff_list)

        # ========== 6. 时频一致性损失 ==========
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

        # ========== 7. 意图路由正则化 ==========
        loss_routing = self.intent_routing(outputs, labels)

        # ========== 8. ✅ 路由掩码约束损失 ==========
        loss_mask = self.mask_loss(outputs)

        # ========== 9. 总损失 ==========
        total = (loss_main +
                 self.w_intent * loss_intent +
                 self.w_contrast * loss_contrast +
                 self.w_sparsity * loss_sparsity +
                 self.w_tf * loss_tf +
                 self.w_routing * loss_routing +
                 self.w_mask * loss_mask)

        # 检查是否为NaN，如果是则返回主损失+意图损失
        if torch.isnan(total):
            return loss_main + self.w_intent * loss_intent

        return total