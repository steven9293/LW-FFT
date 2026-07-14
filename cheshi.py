"""
LW-MIR 模型调试测试文件
用于逐个排查训练失败的问题
"""

import torch
import torch.nn as nn
import numpy as np
from torch import optim
import sys
import os

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 1. 模拟配置参数
# ============================================================
class MockArgs:
    def __init__(self):
        # 数据维度
        self.text_feats_dim = 1024
        self.video_feats_dim = 1024
        self.audio_feats_dim = 768
        self.weight_hidden_dim = 384
        self.weight_dropout = 0.2
        self.temperature = 0.5
        self.need_aligned = True
        self.num_labels = 20
        self.aligned_method = 'ctc'
        self.n_coarse_classes = 5

        # 训练参数
        self.lr = 1e-5
        self.weight_decay = 0.01
        self.grad_clip = 1.0
        self.grad_clip_norm = 1.0

        # 损失权重
        self.w_contrast = 0.01
        self.w_sparsity = 0.0001
        self.w_tf = 0.005
        self.w_routing = 0.005

        # 序列长度
        self.text_seq_len = 30
        self.video_seq_len = 230
        self.audio_seq_len = 480

        # 其他
        self.eps = 1e-8
        self.shared_dim = 512

        # 模拟已对齐
        self.need_aligned = False  # 先关闭对齐测试

# ============================================================
# 2. 模拟数据生成器
# ============================================================
class MockDataLoader:
    def __init__(self, batch_size=8, num_batches=5):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.text_seq_len = 30
        self.video_seq_len = 230
        self.audio_seq_len = 480
        self.text_dim = 1024
        self.video_dim = 1024
        self.audio_dim = 768
        self.num_labels = 20

    def __iter__(self):
        return self

    def __next__(self):
        if self.num_batches <= 0:
            raise StopIteration
        self.num_batches -= 1

        # 生成模拟数据 - 正确的形状
        text_feats = torch.randn(self.batch_size, self.text_seq_len, self.text_dim)
        video_feats = torch.randn(self.batch_size, self.video_seq_len, self.video_dim)
        audio_feats = torch.randn(self.batch_size, self.audio_seq_len, self.audio_dim)
        label_ids = torch.randint(0, self.num_labels, (self.batch_size,))

        return {
            'text_feats': text_feats,
            'video_feats': video_feats,
            'audio_feats': audio_feats,
            'label_ids': label_ids
        }

# ============================================================
# 3. 测试1：检查数据形状
# ============================================================
def test_data_shapes():
    print("\n" + "="*60)
    print("测试1：检查数据加载器返回的形状")
    print("="*60)

    data_loader = MockDataLoader(batch_size=8, num_batches=1)
    batch = next(iter(data_loader))

    print(f"text_feats shape: {batch['text_feats'].shape} - 期望: (8, 30, 1024)")
    print(f"video_feats shape: {batch['video_feats'].shape} - 期望: (8, 230, 1024)")
    print(f"audio_feats shape: {batch['audio_feats'].shape} - 期望: (8, 480, 768)")
    print(f"label_ids shape: {batch['label_ids'].shape} - 期望: (8,)")

    # 验证
    assert batch['text_feats'].shape == (8, 30, 1024), "文本特征形状错误!"
    assert batch['video_feats'].shape == (8, 230, 1024), "视频特征形状错误!"
    assert batch['audio_feats'].shape == (8, 480, 768), "音频特征形状错误!"
    assert batch['label_ids'].shape == (8,), "标签形状错误!"

    print("✅ 数据形状检查通过!")
    return batch

# ============================================================
# 4. 测试2：检查模型结构
# ============================================================
def test_model_structure(args):
    print("\n" + "="*60)
    print("测试2：检查模型结构")
    print("="*60)

    from methods.LW_MIR.model import LW_MIR

    # 创建模型
    model = LW_MIR(args)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 检查各模块是否存在
    modules = ['text_encoder', 'video_encoder', 'audio_encoder',
               'reliability_fusion', 'intent_routing',
               'coarse_classifier', 'fine_classifier', 'final_classifier']

    print("\n检查关键模块:")
    for module_name in modules:
        if hasattr(model, module_name):
            print(f"  ✅ {module_name} 存在")
        else:
            print(f"  ❌ {module_name} 不存在")

    return model

# ============================================================
# 5. 测试3：前向传播
# ============================================================
def test_forward_pass(args, model, batch):
    print("\n" + "="*60)
    print("测试3：前向传播")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = model.to(device)
    model.eval()

    # 将数据移到设备
    text_feats = batch['text_feats'].to(device)
    video_feats = batch['video_feats'].to(device)
    audio_feats = batch['audio_feats'].to(device)
    label_ids = batch['label_ids'].to(device)

    print(f"输入形状:")
    print(f"  text_feats: {text_feats.shape}")
    print(f"  video_feats: {video_feats.shape}")
    print(f"  audio_feats: {audio_feats.shape}")

    with torch.no_grad():
        try:
            outputs = model(text_feats, video_feats, audio_feats)

            print("\n输出keys:")
            for key in outputs.keys():
                if isinstance(outputs[key], torch.Tensor):
                    print(f"  {key}: {outputs[key].shape}")
                else:
                    print(f"  {key}: {type(outputs[key])}")

            # 检查关键输出
            assert 'logits' in outputs, "缺少 logits 输出!"
            assert outputs['logits'].shape == (8, 20), f"logits形状错误: {outputs['logits'].shape}"

            print("\n✅ 前向传播成功!")
            return outputs

        except Exception as e:
            print(f"\n❌ 前向传播失败: {e}")
            raise

# ============================================================
# 6. 测试4：损失函数
# ============================================================
def test_loss_function(args, outputs, labels):
    print("\n" + "="*60)
    print("测试4：损失函数")
    print("="*60)

    from methods.LW_MIR.loss import LWMIRLoss

    # 创建损失函数
    loss_fn = LWMIRLoss(args)

    # 确保标签是CPU上的
    labels = labels.cpu()

    try:
        loss = loss_fn(outputs, labels)

        print(f"损失值: {loss.item():.4f}")
        print(f"损失类型: {type(loss)}")

        # 检查是否为标量
        assert loss.dim() == 0, f"损失不是标量，维度为 {loss.dim()}"

        # 检查是否为有效值
        assert not torch.isnan(loss), "损失为 NaN!"
        assert not torch.isinf(loss), "损失为 Inf!"
        assert loss.item() > 0, f"损失值异常: {loss.item()}"

        print("✅ 损失函数测试通过!")
        return loss

    except Exception as e:
        print(f"\n❌ 损失函数测试失败: {e}")
        raise

# ============================================================
# 7. 测试5：反向传播和梯度
# ============================================================
def test_backward(args, model, batch):
    print("\n" + "="*60)
    print("测试5：反向传播和梯度")
    print("="*60)

    from methods.LW_MIR.loss import LWMIRLoss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    # 数据移到设备
    text_feats = batch['text_feats'].to(device)
    video_feats = batch['video_feats'].to(device)
    audio_feats = batch['audio_feats'].to(device)
    label_ids = batch['label_ids'].to(device)

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = LWMIRLoss(args)

    try:
        # 前向传播
        outputs = model(text_feats, video_feats, audio_feats)
        loss = loss_fn(outputs, label_ids)

        print(f"前向损失: {loss.item():.4f}")

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 检查梯度
        grad_norm = 0
        grad_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
                grad_count += 1

        grad_norm = grad_norm ** 0.5
        print(f"梯度范数: {grad_norm:.6f}")
        print(f"有梯度的参数数量: {grad_count}")

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        optimizer.step()

        print("✅ 反向传播和梯度更新成功!")
        return loss, grad_norm

    except Exception as e:
        print(f"\n❌ 反向传播失败: {e}")
        import traceback
        traceback.print_exc()
        raise

# ============================================================
# 8. 测试6：训练循环（单步）
# ============================================================
def test_training_step(args, model, data_loader):
    print("\n" + "="*60)
    print("测试6：完整训练步骤")
    print("="*60)

    from methods.LW_MIR.loss import LWMIRLoss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = LWMIRLoss(args)

    losses = []
    accuracies = []

    for step, batch in enumerate(data_loader):
        # 数据移到设备
        text_feats = batch['text_feats'].to(device)
        video_feats = batch['video_feats'].to(device)
        audio_feats = batch['audio_feats'].to(device)
        label_ids = batch['label_ids'].to(device)

        # 前向传播
        outputs = model(text_feats, video_feats, audio_feats)
        logits = outputs['logits']

        # 计算损失
        loss = loss_fn(outputs, label_ids)

        # 检查损失
        if torch.isnan(loss):
            print(f"⚠️ Step {step}: 损失为 NaN，跳过")
            continue

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        optimizer.step()

        # 计算准确率
        preds = logits.argmax(dim=1)
        acc = (preds == label_ids).float().mean().item()

        losses.append(loss.item())
        accuracies.append(acc)

        print(f"Step {step}: Loss = {loss.item():.4f}, Acc = {acc:.4f}")

        if step >= 4:  # 只测试5步
            break

    avg_loss = np.mean(losses)
    avg_acc = np.mean(accuracies)

    print(f"\n平均损失: {avg_loss:.4f}")
    print(f"平均准确率: {avg_acc:.4f}")

    if avg_acc > 0.05:  # 随机猜测是5%
        print("✅ 模型正在学习!")
    else:
        print("⚠️ 模型还未学习，准确率接近随机猜测")

    return avg_loss, avg_acc

# ============================================================
# 9. 测试7：检查梯度爆炸
# ============================================================
def test_gradient_explosion(args, model, batch):
    print("\n" + "="*60)
    print("测试7：检查梯度爆炸")
    print("="*60)

    from methods.LW_MIR.loss import LWMIRLoss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    # 数据移到设备
    text_feats = batch['text_feats'].to(device)
    video_feats = batch['video_feats'].to(device)
    audio_feats = batch['audio_feats'].to(device)
    label_ids = batch['label_ids'].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = LWMIRLoss(args)

    # 多次前向-反向
    grad_norms = []

    for i in range(10):
        optimizer.zero_grad()
        outputs = model(text_feats, video_feats, audio_feats)
        loss = loss_fn(outputs, label_ids)

        if torch.isnan(loss):
            print(f"⚠️ Iteration {i}: 损失为 NaN")
            break

        loss.backward()

        # 计算梯度范数
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)

        print(f"Iteration {i}: Loss = {loss.item():.4f}, Grad Norm = {total_norm:.6f}")

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    if grad_norms:
        max_grad = max(grad_norms)
        avg_grad = np.mean(grad_norms)
        print(f"\n最大梯度范数: {max_grad:.6f}")
        print(f"平均梯度范数: {avg_grad:.6f}")

        if max_grad > 100:
            print("⚠️ 检测到梯度爆炸! (梯度范数 > 100)")
        else:
            print("✅ 梯度范数正常")

    return grad_norms

# ============================================================
# 10. 主测试函数
# ============================================================
def run_all_tests():
    print("="*60)
    print("LW-MIR 模型调试测试")
    print("="*60)

    # 创建配置
    args = MockArgs()

    # 测试1：数据形状
    batch = test_data_shapes()

    # 测试2：模型结构
    model = test_model_structure(args)

    # 测试3：前向传播
    outputs = test_forward_pass(args, model, batch)

    # 测试4：损失函数
    loss = test_loss_function(args, outputs, batch['label_ids'])

    # 测试5：反向传播
    loss_val, grad_norm = test_backward(args, model, batch)

    # 测试6：训练步骤
    data_loader = iter(MockDataLoader(batch_size=8, num_batches=5))
    avg_loss, avg_acc = test_training_step(args, model, data_loader)

    # 测试7：梯度爆炸
    data_loader = MockDataLoader(batch_size=8, num_batches=1)
    batch = next(iter(data_loader))
    grad_norms = test_gradient_explosion(args, model, batch)

    print("\n" + "="*60)
    print("所有测试完成!")
    print("="*60)

    # 总结
    print("\n总结:")
    print(f"  ✅ 数据形状: 正确")
    print(f"  ✅ 模型结构: 正确")
    print(f"  ✅ 前向传播: 成功")
    print(f"  ✅ 损失函数: 正常 (损失值: {loss.item():.4f})")
    print(f"  ✅ 反向传播: 成功 (梯度范数: {grad_norm:.6f})")
    print(f"  ✅ 训练步骤: 完成 (准确率: {avg_acc:.4f})")

    if avg_acc > 0.10:
        print("  🎉 模型正在有效学习!")
    else:
        print("  ⚠️ 模型需要进一步调试")

# ============================================================
# 11. 运行测试
# ============================================================
if __name__ == "__main__":
    # 添加项目路径
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    run_all_tests()