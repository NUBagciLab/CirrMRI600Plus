# 🧠 Large Scale MRI Collection and Segmentation of Cirrhotic Liver

This repository contains the **official implementation** for all baseline models used to benchmark our medical imaging dataset **CirrMRI600+**. It includes training pipelines, model configurations, and preprocessing scripts for 3D liver MRI segmentation tasks.

---

## 📌 Overview

We provide implementations and training scripts for several popular and effective models in 3D medical image segmentation:

- ✅ Custom: **SynergyNet3D**
- ✅ MONAI-based: `AttentionUnet`, `SwinUNETR`, `UNet`
- ✅ Official Implementations: `nnUNet`, `nnFormer`, `MedSegDiff`

---

## 🧪 Baseline Models

### 🔹 SynergyNet3D (Custom Implementation)
- Transformer-augmented 3D U-Net.
- Multi-resolution ROI processing.
- Implemented in `model/trans_3DUnet.py`.
- Trained using `train_lin.py`.

### 🔹 MONAI Baselines
Implemented with [MONAI](https://monai.io/):
- `UNet`
- `AttentionUnet`
- `SwinUNETR`

### 🔹 Official Repositories (External)
Used via their own official codebases:
- [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- [nnFormer](https://github.com/282857341/nnFormer)
- [MedSegDiff](https://github.com/OpenGVLab/MedSegDiff)

---

## 📦 Installation

Create a new Python environment (optional) and install dependencies:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```txt
torch>=1.10
monai>=1.0.1
nibabel
numpy
scipy
tqdm
```

---

## 📥 Download Dataset

The **CirrMRI600+** dataset is available for download:

➡️ [Download CirrMRI600+ Dataset](https://osf.io/cuk24/)  
📦 Size: ~15 GB  
📝 Format: NIfTI (.nii.gz)

---

## 📁 Dataset Structure

Make sure your dataset is organized in the following format:

```
/data/dataset_path/
├── images/
│   ├── sample_1.nii.gz
│   ├── sample_2.nii.gz
│   └── ...
└── labels/
    ├── sample_1.nii.gz
    ├── sample_2.nii.gz
    └── ...
```

---

## 🔄 Preprocessing Pipeline

Each `.nii.gz` volume undergoes the following steps:

1. **Loading** using `nibabel`
2. **Conversion** to PyTorch tensors
3. **Resizing** to `(256, 256, 80)` using:
   ```python
   monai.transforms.spatial.functional.resize(...)
   ```
4. **Z-score Normalization**:
   ```python
   image = (image - torch.mean(image)) / torch.std(image)
   ```

---

## 🔁 Postprocessing

After model prediction:

- Apply sigmoid activation:
  ```python
  probs = torch.sigmoid(output)
  ```

- Optional thresholding:
  ```python
  binary_output = (probs > 0.5).float()
  ```

- Further postprocessing (e.g., connected components) can be added based on task requirements.

---

## 🏋️‍♀️ Training SynergyNet3D

To start training:

```bash
python train_lin.py
```

### 🔧 Training Configuration

- **Model**: MaskTransUnet (SynergyNet3D)
- **Loss**: Dice + Binary Cross Entropy (`DiceBCELoss`)
- **Optimizer**: AdamW
- **Learning Rate**: `1e-4`
- **Batch Size**: `4`
- **Epochs**: `1000`
- **Checkpoint**: Saved every 5 epochs to `/datadrive/pan_dataset/zheyuan_model/`

---

## 📊 Evaluation

Evaluation and metrics scripts will be released in future updates. You may evaluate with:

- Dice Coefficient
- Hausdorff Distance
- Volume Overlap

---

## 🧾 Citation

If you use this dataset or code, please cite:

```bibtex
@article{jha2024cirrmri600+,
  title={Large Scale MRI Collection and Segmentation of Cirrhotic Liver},
  author={Jha, Debesh and Susladkar, Onkar Kishor and Gorade, Vandan and Keles, Elif and Antalek, Matthew and Seyithanoglu, Deniz and Cebeci, Timurhan and Aktas, Halil Ertugrul and Kartal, Gulbiz Dagoglu and Kaymakoglu, Sabahattin and others},
  journal={Nature Scientific Data},
  year={2025}
}
```

---

## 📝 License

This repository is released under the [MIT License](LICENSE).
