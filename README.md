# CVM-AI-Classifier
An AI-driven system for automated cervical vertebral maturation (CVM) stage classification. 

## Overview
This project implements a deep learning-based system for automatically classifying cervical vertebral maturation stages from lateral cephalometric radiographs. The system provides accurate CVM stage predictions with visualization capabilities through Grad-CAM heatmaps and confidence scores for clinical interpretation.
## Dataset
This project uses the **CVM-900** dataset. For more details on the dataset, including its citation, please refer to the [DATASET.md](./DATASET.md) file.
## Features
- Automated CVM stage classification (CS1-CS6)
- Interactive web interface for image analysis
- Grad-CAM heatmap visualization
- Confidence scoring for predictions
- Combined view with heatmap overlay
## Installation and Usage
### 1. Clone the repository:
```bash
git clone https://github.com/yourusername/CVM-AI-Classifier.git
cd CVM-AI-Classifier
```

### 2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the application:
```bash
python main.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
