# 🧑‍👩‍👧‍👦 Age & Gender Prediction with ResNet50

A deep learning application that predicts age and gender from facial images using a ResNet50 model trained on the UTK Face dataset. The project includes both the training pipeline and a user-friendly Streamlit web interface for real-time predictions.

## 🌟 Features

- **Multi-task Learning**: Simultaneously predicts age (regression) and gender (classification)
- **High Accuracy**: Achieves ~90% gender classification accuracy and ~6 years age prediction MAE
- **Transfer Learning**: Leverages pre-trained ResNet50 for efficient training
- **Interactive Web App**: Streamlit-based interface for easy image upload and prediction
- **Real-time Inference**: Fast predictions with confidence scores
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Gender Accuracy | ~90% |
| Age MAE | ~6 years |
| Architecture | ResNet50 + Custom Heads |
| Dataset | UTK Face (20,000+ images) |
| Training Time | ~15 epochs |

## 🚀 Quick Start

### Prerequisites

- Python 3.8-3.11
- Anaconda/Miniconda (recommended)
- CUDA-compatible GPU (optional, for faster training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/age-gender-prediction.git
cd age-gender-prediction
```

2. **Create conda environment**
```bash
conda create -n age-gender-app python=3.11
conda activate age-gender-app
```

3. **Install dependencies**
```bash
conda install tensorflow numpy pandas matplotlib seaborn scikit-learn
pip install streamlit pillow opencv-python-headless
```

4. **Run the Streamlit app**
```bash
streamlit run app.py
```

5. **Open your browser** and go to `http://localhost:8501`

## 📁 Project Structure

```
age-gender-prediction/
├── app.py                          # Streamlit web application
├── utk_face_resnet50_90_accuracy.py # Model training script
├── README.md                      # This file
└── assets/                        # Screenshots and demo images
    ├── image1.jpg
    ├── image2.jpg
    ├── image3.jpg
    └── image4.jpg
```

## 🔧 Usage

### Training the Model

1. **Download the UTK Face dataset**
   - Visit [UTK Face Dataset](https://susanqq.github.io/UTKFace/)
   - Extract to `/kaggle/input/utk-face-cropped/utkcropped/` or update the path in the script
  
2. **Download the ResNet50 Model**
   - Visit [UTK Face Dataset]([https://susanqq.github.io/UTKFace/](https://www.kaggle.com/code/abdelrahmantarekm/utk-face-resnet50))t

3. **Run the training script**
```bash
python utk_face_resnet50_90_accuracy.py
```

4. **Monitor training progress**
   - The script will output training metrics and save the best model
   - Training visualizations will be displayed automatically

### Using the Web App

1. **Start the application**
```bash
streamlit run app.py
```

2. **Upload an image**
   - Support formats: JPG, JPEG, PNG
   - Best results with clear, front-facing portraits

3. **View predictions**
   - Gender classification with confidence score
   - Age prediction with estimated range
   - Visual confidence indicators

## 🏗️ Model Architecture

The model uses a **multi-task learning** approach with ResNet50 as the backbone:

```
Input (256x256x3)
       ↓
   ResNet50 Base
   (Pre-trained)
       ↓
 Global Average Pooling
       ↓
     Split into two heads:
       
Gender Head:          Age Head:
Dense(256) + ReLU     Dense(256) + ReLU
BatchNorm             BatchNorm
Dropout(0.5)          Dropout(0.5)
Dense(1) + Sigmoid    Dense(1) + Linear
```

### Key Features:
- **Transfer Learning**: Pre-trained ImageNet weights
- **Fine-tuning**: Last 20 layers trainable
- **Data Augmentation**: Random brightness and horizontal flip
- **Class Balancing**: Weighted loss for gender imbalance
- **Multi-task Optimization**: Combined loss function

## 📈 Training Details

### Data Preprocessing
- Image resizing to 256x256 pixels
- Normalization to [0,1] range
- Random augmentation for training
- Train/Validation/Test split: 60%/20%/20%

### Training Configuration
- **Optimizer**: Adam (lr=1e-4)
- **Loss Functions**: 
  - Gender: Binary Cross-entropy
  - Age: Mean Absolute Error
- **Batch Size**: 32
- **Epochs**: 15
- **Callbacks**: Model checkpointing for best validation loss

## 🎯 Results & Evaluation

The model provides comprehensive evaluation metrics:

### Gender Classification
- **Accuracy**: ~90%
- **Precision/Recall**: Balanced performance
- **Confusion Matrix**: Detailed classification breakdown

### Age Prediction
- **MAE**: ~6 years average error
- **Error Distribution**: Most predictions within ±10 years
- **Age Range Performance**: Better accuracy for middle-aged subjects

### Visualization Features
- Confusion matrix heatmaps
- Error distribution histograms
- Sample predictions with ground truth
- Training history plots

## 🌐 Web Application Features

### User Interface
- **Clean, intuitive design** with responsive layout
- **Real-time predictions** with loading indicators
- **Confidence visualization** with progress bars
- **Batch processing** support for multiple images

### Technical Features
- **Efficient caching** for model loading
- **Error handling** with user-friendly messages
- **Image preprocessing** pipeline
- **Mobile-responsive** design

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

## 🔬 Technical Specifications

### Model Details
- **Framework**: TensorFlow/Keras
- **Architecture**: ResNet50 + Custom Layers
- **Input Size**: 256×256×3
- **Parameters**: ~23.5M (ResNet50)
- **Model Size**: ~300MB

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **UTK Face Dataset** creators for providing the training data
- **ResNet50** architecture by He et al.
- **Streamlit** team for the amazing web framework
- **TensorFlow** community for the deep learning framework

---

⭐ If you found this project helpful, please give it a star!
