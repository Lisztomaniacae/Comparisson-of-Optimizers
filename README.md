# 🖥️ MNIST Classification with PyTorch & YAML

## 📌 Project Overview
This repository provides a **fully configurable PyTorch implementation** of a Convolutional Neural Network (CNN) for **handwritten digit classification** using the MNIST dataset. It leverages a **YAML configuration file** for defining all hyperparameters, making the setup flexible and easy to modify without altering the code.

## 🚀 Key Features
- ✅ **Modular YAML Configuration** for dataset, model, training, and optimizer settings.
- ✅ **Multiple Optimizers Supported**: Adam, Adadelta, and SGD.
- ✅ **Batch Normalization & Configurable Model Architecture**.
- ✅ **Learning Rate Scheduling**: Supports ExponentialLR and StepLR.
- ✅ **Automatic Training Visualization** (if enabled in `config.yaml`).
- ✅ **Device Agnostic**: Runs on **MPS, CUDA, or CPU** seamlessly.

---
## 🛠️ Installation & Setup

### 1️⃣ Install Dependencies
Ensure Python and pip are installed, then run:

```bash
pip install torch torchvision tqdm matplotlib pyyaml
```

### 2️⃣ Clone the Repository
```bash
git clone https://github.com/your-repo/MNIST-YAML.git
cd MNIST-YAML
```

### 3️⃣ Run the Jupyter Notebook
```bash
jupyter notebook MNIST_Project_with_YAML.ipynb
```

---
## ⚙️ Configuration File (`config.yaml`)
All **hyperparameters** and **training settings** are stored in `config.yaml`, allowing modifications **without editing the code**.

### Example Configuration:
```yaml
optimizer:
  optimizers:
    - "Adam"
    - "Adadelta"
    - "SGD"
  learning_rates:
    Adam: 0.0005
    Adadelta: 1.0
    SGD: 0.05
  momentum: 0.0
```
🔹 **To change the optimizer**, update `optimizer.optimizers` in `config.yaml`.

---
## 📊 Model Architecture
The CNN follows a structured architecture for **efficient digit classification**:
- **2 Convolutional Layers** (16, 32 filters)
- **Batch Normalization** for stable training (configurable in `config.yaml`)
- **Fully Connected Layers**: 128 neurons → 10 output classes
- **ReLU Activation & Max Pooling** for feature extraction
- **Softmax Output** for classification

---
## 📈 Training Visualization
If `visualize_training` is enabled in `config.yaml`, the notebook will generate performance plots:
- 📉 **Training vs Testing Loss**
- 📊 **Training vs Testing Accuracy**

---
## ⚡ Optimizer Comparison
This project allows easy comparison of **Adam, Adadelta, and SGD**. All selected optimizers in `config.yaml` will be run sequentially.

💡 **Modify `config.yaml` to switch between optimizers effortlessly!**

---
## 📂 Repository Files
```
├── MNIST_Project_with_YAML.ipynb  # Jupyter Notebook implementation
├── config.yaml                     # Configurable settings
```

---
## 📌 Running with Custom Configurations
To train using **custom settings**, edit `config.yaml` and re-run:
```bash
jupyter notebook MNIST_Project_with_YAML.ipynb
```

---
## 📜 License
📄 MIT License - Free to use, modify, and distribute!

---
## 📬 Contributions & Support
Have feedback or improvements? Feel free to submit a **Pull Request** or open an issue on **GitHub**! 🎉
