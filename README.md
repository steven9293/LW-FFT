# Learnable Wavelet-FFT Dual-Path Reasoning for Multimodal Intent Recognition

This is the official PyTorch implementation of our paper: 
<br>
<br>
**"Learnable Wavelet-FFT Dual-Path Reasoning for Multimodal Intent Recognition"**.

## 1. Introduction
Multimodal intent recognition (MIR), which leverages linguistic, acoustic, and visual cues to identify user intents. However,existing MIR methods mainly capture local semantic patterns and temporal dependencies, while the complementary frequency-domain cues across modalities remain underexplored. To address these limitations, this paper proposes a learnable Wavelet-FFT dual-path inference framework (LW-FFT).Furthermore, we design an intent-aware dimension mask routing mechanism to dynamically filter redundant multimodal noise and align features with specific intents, enabling a precise extraction of deep multimodal semantic cues for accurate recognition.

## 2. Dependencies

We use Anaconda to create the Python environment:

```bash
conda create --name LW_FFT python=3.12 -y
conda activate LW_FFT
```
Install all required libraries:

```bash
pip install -r requirements.txt
```
## 3. Usage

### 3.1 Dataset Preparation

You can evaluate the performance of our proposed LW-FFT on [MIntRec](https://github.com/thuiar/MIntRec) and [MIntRec 2.0](https://github.com/thuiar/MIntRec2.0) by downloading the datasets and placing them in the `data/` directory. 

Ensure the directory structure looks like this:

```text
LW-FFT/
├── data/
│   ├── MIntRec/
│   └── MIntRec2.0/
...
```
### 3.2 Training & Testing

* MIntRec
```bash
sh examples/run_LW_FFT_MIntRec.sh
```
* MIntRec2.0
```bash
sh examples/run_LW_FFT_MIntRec2.sh
```

## 4. Model
Overall architecture diagram of the LW-FFT：

![LW-FFT Architecture](figs/architecture.png)

Implementation Details of the Time-Frequency Sequence Encode：

![LW-FFT Architecture](figs/Time_Fre.png)

## 5. Experimental Results

![Experimental Results](figs/Experimental_Results.png)

## 6. Code Release & Maintenance

We are committed to promoting reproducible research. The current repository provides the **basic training pipeline, data processing scripts, and configuration files** for the LW-FFT framework to facilitate the peer-review process.

To protect the intellectual property of our core innovations (specifically the **Learnable Wavelet Layer** and the **Intent-Aware Dimension Mask Routing module**), the implementations of these specific modules are currently abstracted. 

**The complete source code, including all core modules and pre-trained weights, will be fully open-sourced immediately upon the acceptance of this paper.**

For any inquiries or requests regarding the code, please feel free to contact the authors.

## 7. Acknowledgments

Some of the codes in this repo are adapted from [MIntRec](https://github.com/thuiar/MIntRec), and we are greatly thankful.

If you have any questions, please open issues and illustrate your problems as detailed as possible.

