# LW-FFT

This is the official PyTorch implementation of our paper: **"Learnable Wavelet-FFT Dual-Path Reasoning for Multimodal Intent
Recognition"**.

## 📖 Introduction
Multimodal intent recognition (MIR), which leverages linguistic, acoustic, and visual cues to identify user intents, plays a central role in advancing modern interactive dialogue systems. However,
existing MIR methods mainly capture local semantic patterns and temporal dependencies, while the
complementary frequency-domain cues across modalities remain underexplored. To address these
limitations, this paper proposes a learnable Wavelet-FFT dual-path inference framework (LW-FFT).
This framework couples FFT-based global spectral analysis with an adaptive time-frequency module,
where the former captures global frequency-domain regularities and the latter extracts local fine-
grained features via learnable wavelet transforms. Furthermore, we design an intent-aware dimension
mask routing mechanism to dynamically filter redundant multimodal noise and align features with spe-
cific intents. 

## 🛠️ Requirements
Please install the required packages using pip:
```bash
pip install -r requirements.txt
