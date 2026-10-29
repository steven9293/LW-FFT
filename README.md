# Learnable Wavelet-FFT Dual-Path Reasoning for Multimodal Intent Recognition

This is the official PyTorch implementation of our paper: 
**"Learnable Wavelet-FFT Dual-Path Reasoning for Multimodal Intent Recognition"**.

## 📖 Introduction
Multimodal intent recognition (MIR), which leverages linguistic, acoustic, and visual cues to identify user intents. However,existing MIR methods mainly capture local semantic patterns and temporal dependencies, while the complementary frequency-domain cues across modalities remain underexplored. To address these limitations, this paper proposes a learnable Wavelet-FFT dual-path inference framework (LW-FFT).Furthermore, we design an intent-aware dimension mask routing mechanism to dynamically filter redundant multimodal noise and align features with specific intents, enabling a precise extraction of deep multimodal semantic cues for accurate recognition.

## 🛠️ Requirements
Please install the required packages using pip:
```bash
pip install -r requirements.txt
