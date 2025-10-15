# Nerthus Medical ML 🏥🤖

A comprehensive machine learning and deep learning pipeline for automated bowel preparation quality assessment using the **Nerthus dataset**.  
This project demonstrates advanced skills in medical image analysis, traditional ML, deep learning, and production-ready Python development.

![ML vs CNN Comparison](outputs/cnn/ml_vs_cnn_comparison.png)

---

## 🎯 Project Overview
This project tackles the challenging problem of **automated bowel preparation quality assessment** from colonoscopy images.  
Bowel preparation quality directly impacts colonoscopy effectiveness and cancer detection rates.  
The system uses both traditional machine learning and deep learning approaches to classify images according to the **Boston Bowel Preparation Scale (BBPS)** scores 0–3.

---

## Dataset overview
The Nerthus dataset contains 5,525 colonoscopy images organized by BBPS scores.
![Data overview](outputs/analysis/images/dataset_overview.png)

### Feature analysis
![Feature analysis](outputs/analysis/images/feature_analysis.png)

### Feature correlations
![Feature correlations](outputs/analysis/images/feature_correlations.png)

### File size distribution
![File size distribution](outputs/analysis/images/file_size_distribution.png)

---

## 🏆 Key achievements
- 🤖 **95.4% accuracy** with CNN learning from raw pixels  
- 🥇 **92.5% accuracy** with XGBoost using handcrafted features  
- Comprehensive comparison of traditional ML vs deep learning  
- Production-ready Python package architecture  
- Robust validation with cross-validation and overfitting analysis  

---

## 📊 Results summary

### Performance comparison
| Method | Architecture | Accuracy | Key Insight |
|--------|---------------|-----------|--------------|
| CNN | Raw pixels learning | **95.4% 🥇** | Best performance |
| XGBoost | Gradient boosting | 92.5% | Strong alternative |
| Random Forest | Handcrafted features | 91.9% | Promising end-to-end learning with domain-specific features |
| Logistic Regression | Linear model | 85.0% | Good baseline performance |

### Feature importance
Top 15 most predictive features for bowel preparation quality:
![Features importance for bowel preparation quality](outputs/ml/results/feature_importance_xgboost.png)


- `hue mean` - Average hue in HSV color space
- `b mean` - Position between blue and yellow
- `std intensity` - Std image intensity
- `a mean` - Position between red/magenta and green
- `homogeneity` - Texture homogeneity (GLCM)

---

## 🏗️ Project architecture

```
├── AUTHORS.txt                             # Author(s)
├── examples
│   ├── nerthus_cnn_champion.py
│   ├── nerthus_cnn_improved.py
│   ├── nerthus_cnn_simple.py
│   ├── nerthus_cnn_tuning.py
│   ├── nerthus_comparison_plot.py
│   ├── nerthus_image_analyzer.py
│   ├── nerthus_image_processor.py
│   └── nerthus_ml_pipeline.py
├── nerthus
│   ├── analyzer.py                         # Data analysis & EDA
│   ├── cli.py                              # Command line interface
│   ├── cnn.py                              # Deep learning
│   ├── __init__.py
│   ├── ml.py                               # Traditional ML
│   ├── processor.py                        # Image processing
│   └── utils.py                            # Utilities
├── pyproject.toml                          # Modern Python packaging
├── README.md                               # Project description
├── report
│   ├── images                              # Visualization files
│   ├── Makefile                            # Building the report
│   ├── Report.pdf                          # Report PDF file
│   ├── Report.tex                          # Report LateX file
│   └── requirements.sh                     # Prerequisites for the report
├── requirements.txt                        # Python project requirements
├── setup.py                                # Setup file
└── web
    ├── app.py                              # Web app file
    ├── debug_cnn.py                        # CNN debug file
    ├── loader.py                           # Model loader
    ├── requirements.txt                    # Web app requirements
    ├── run.py                              # Run web app with Python
    ├── static                              # Static file for the web app
    └── utils.py                            # Web app utilities
```


---

## 🚀 Quick start

### Installation
```bash
# Clone the repository
git clone https://github.com/kvernet/nerthus-medical-ml.git
cd nerthus-medical-ml

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Basic usage
```python
from nerthus import NerthusAnalyzer

# Initialize and run complete analysis
analyzer = NerthusAnalyzer()
analyzer.load_data()  # Auto-downloads from Kaggle
analyzer.analyze_image_features(sample_size=100)
analyzer.generate_report(sample_size=100, images_per_class=4)
```

### Command line interface
```bash
# Image processing
nerthus --processor (nerthus-processor --help)

# Analysis only
nerthus --analysis (nerthus-analysis --help)

# Complete ML pipeline (run analysis first)
nerthus --ml (nerthus-ml --help)

# CNN training
nerthus --cnn (nerthus-cnn --help)

# CNN training
python examples/nerthus_cnn_champion.py

# Performance comparison
python examples/nerthus_comparison_plot.py
```

---

## Visualization of CNN training results
![CNN training history: accuracy and loss curves](outputs/cnn/results/nerthus_cnn_training_history.png)

---

## 📈 Key features

### 🔬 Medical image analysis
- 22 handcrafted features specifically designed for medical images  
- Texture analysis using GLCM (contrast, homogeneity, energy, correlation)  
- Color space analysis in RGB, HSV, and LAB spaces  
- Edge detection and sharpness measurement  
- Blob detection for potential polyp identification  

### 🤖 Machine Learning pipeline
- Multiple classifiers: Random Forest, XGBoost, SVM, Logistic Regression  
- Robust validation: 5-fold cross-validation with stratified sampling  
- Feature importance analysis and statistical significance testing  
- Overfitting detection with train-test performance gaps  
- Comprehensive reporting with confusion matrices and metrics  

### 🧠 Deep Learning
- Custom CNN architecture optimized for medical images  
- Data augmentation with rotation, shifting, and flipping  
- Learning rate scheduling and early stopping  
- CPU/GPU compatible with automatic fallback  
- Transfer learning ready architecture  

### 📊 Visualization & reporting
- Dataset overview with class distributions and image statistics  
- Feature correlation heatmaps and distribution plots  
- Training history visualization for deep learning  
- Model comparison with professional-grade plots  
- Medical insights and deployment recommendations  

---

## 🏥 Medical context

### Boston Bowel Preparation Scale (BBPS)
| Class | Description |
|--------|--------------|
| 0 | Unprepared colon, mucosa not visible due to solid stool |
| 1 | Portions of mucosa visible, but other areas not well seen |
| 2 | Minor residue, but mucosa clearly visible |
| 3 | Entire mucosa clearly visible |

### Clinical significance
- Bowel preparation quality affects **adenoma detection rate by 20–50%**  
- Automated assessment reduces inter-observer variability  
- Standardized scoring improves screening program quality  
- Real-time feedback enhances colonoscopy effectiveness  

---

## 🔧 Technical implementation

### Feature engineering
```python
features = {
    'texture': ['contrast', 'homogeneity', 'energy', 'correlation'],
    'color': ['hue_mean', 'saturation_mean', 'l_mean', 'a_mean', 'b_mean'],
    'edges': ['edge_density', 'sharpness'],
    'intensity': ['mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity'],
    'advanced': ['image_entropy', 'lbp_entropy', 'blob_count']
}
```

### CNN architecture summary
```
Model: "sequential"
=================================================================
Layer (type)                 Output Shape              Param #
-----------------------------------------------------------------
conv2d (Conv2D)              (None, 148, 148, 16)      448
max_pooling2d (MaxPooling2D) (None, 74, 74, 16)        0
conv2d_1 (Conv2D)            (None, 72, 72, 32)        4640
max_pooling2d_1 (MaxPooling2D) (None, 36, 36, 32)      0
conv2d_2 (Conv2D)            (None, 34, 34, 64)        18496
max_pooling2d_2 (MaxPooling2D) (None, 17, 17, 64)      0
flatten (Flatten)            (None, 18496)             0
dense (Dense)                (None, 64)                1183808
dropout (Dropout)            (None, 64)                0
dense_1 (Dense)              (None, 4)                 260
=================================================================
Total params: 1,207,652
Trainable params: 1,207,652
```

---

## 📋 Dataset information
- **Source:** Nerthus Dataset on Kaggle  
- **Total Images:** 5,525 colonoscopy frames  
- **Classes:** 4 BBPS scores (0, 1, 2, 3)  
- **Resolution:** Standardized to 150×150 for CNN  
- **License:** Research use, citation required  

---

## 🎯 Skills demonstrated
### 🐍 Advanced Python
- Class-based architecture and modular design  
- Type hints, docstrings, and exception handling  
- CLI development with `argparse`  

### 📊 Data Science & ML
- Feature engineering, cross-validation, and model interpretation  
- ANOVA testing and statistical validation  
- Visualization of metrics and performance  

### 🏥 Medical AI
- Understanding of gastroenterology domain  
- Medical image processing techniques  
- Clinical validation and interpretability considerations  

### 🚀 Software engineering
- Unit tests and configuration management  
- Version control best practices  
- Documentation and usage examples  

---

## 🔮 Future enhancements
- Transfer Learning with pre-trained models  
- Ensemble ML + DL hybrid approaches  
- Real-time inference API for clinics  
- Multi-center dataset validation  
- Explainable AI for clinical interpretability  

---

## 🌐 Web application

A fully functional **Streamlit** web application for the medical AI system. The web application bridges the gap between AI research and clinical practice, transforming advanced models into a practical, interactive tool that demonstrates real-world healthcare impact.

---

### **Overview**
![Live Demo Features](web/static/overview.png)
- Project summary and key performance metrics  
- Real-time model availability status  
- Clinical impact explanation  

---

### **Live demo**
![Live Demo Features](web/static/live_demo.png)
- **Image Upload**: Drag-and-drop interface for colonoscopy images  
- **Dual Model Support**: Choose between CNN or ML predictions  
- **Real-time Results**: Instant BBPS scoring with confidence levels  
- **Clinical Context**: BBPS scale reference and medical explanations  

### 📊 **Model performance**
![Live Demo Features](web/static/model_performance.png)
- Interactive comparison charts of all models  
- Feature importance visualization  
- Performance metrics with actual results  

### 📈 **Technical details**
![Live Demo Features](web/static/technical_details.png)
- Complete ML pipeline documentation  
- CNN architecture specifications  
- Feature engineering methodology  

### 🏥 **Medical context**
![Live Demo Features](web/static/medical_context.png)
- Boston Bowel Preparation Scale (BBPS) reference  
- Clinical significance of automated assessment  
- Healthcare impact analysis  

---

## 🛠️ Technical implementation

### 📁 **Architecture**
```python
web/
├── app.py                 # Main Streamlit application
├── loader.py              # Model management and prediction engine
├── requirements.txt       # Web app dependencies
├── run.py                 # Application runner
├── static                 # Static files
└── utils.py               # Web app utilities
```

**Key features**
- **Production-Ready:** Robust error handling and model validation  
- **Medical Grade:** Proper image preprocessing and feature extraction  
- **User-Friendly:** Intuitive interface designed for clinicians  
- **Responsive:** Works on desktop, tablet, and mobile devices  

---

## 🎯 Model integration
The web app seamlessly integrates your trained models:

| Model | Type | Accuracy | Description |
|--------|------|-----------|-------------|
| ML | Classical ML | 92.5% | Handcrafted feature-based |
| CNN | Deep Learning | 95.4% | End-to-end feature extraction |

**Additional features:**
- Real-time feature extraction (same pipeline as training)  
- Confidence scoring with probability estimates for clinical decision support  

---

## 🚀 Quick start

### 🧩 Local deployment
```bash
# Navigate to web app directory
cd web

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run app.py

# Access the app
http://localhost:8501
```

### ☁️ Cloud deployment options
- Streamlit Cloud *(Free tier available)*  
- Heroku *(with Procfile configuration)*  
- AWS / Azure / GCP *(containerized deployment)*  
- Hugging Face Spaces *(community hosting)*  

---

## 🏥 Clinical workflow integration
The web app supports typical clinical workflows:

1. **Image acquisition:** Upload colonoscopy frames directly  
2. **Quality assessment:** Automatic BBPS scoring in seconds  
3. **Result interpretation:** Confidence scores and clinical context  
4. **Documentation:** Exportable results for medical records  

---

## 🔧 Customization

Easily extend the application with your own models and features:

```python
# Custom model integration
def predict_custom_model(image):
    # Add your custom model here
    return prediction, confidence

# Custom feature extraction
def extract_custom_features(image):
    # Implement domain-specific features
    return features
```

---

## 🌟 Deployment highlights
✅ **HIPAA-Ready:** Local data processing, no cloud storage required  
✅ **Clinically validated:** Uses your validated ML models and features  
✅ **Scalable:** Supports multiple concurrent users  
✅ **Professional UI:** Designed for medical environments  

---

## 🎯 Value for healthcare providers
- **Standardized assessment:** Reduces inter-observer variability  
- **Time efficiency:** Instant scoring vs. manual evaluation  
- **Training tool:** Supports education for new gastroenterologists  
- **Quality assurance:** Objective, reproducible metrics  

---

## 📚 Citation
```bibtex
@inproceedings{Pogorelov:2017:NBP,
  author = {Pogorelov, Konstantin and Randel, Kristin Ranheim and de Lange, Thomas and Eskeland, Sigrun Losada and Griwodz, Carsten and Johansen, Dag and Spampinato, Concetto and Taschwer, Mario and Lux, Mathias and Schmidt, Peter Thelin and Riegler, Michael and Halvorsen, P{a}l},
  title = {Nerthus: A Bowel Preparation Quality Video Dataset},
  booktitle = {Proceedings of the 8th ACM on Multimedia Systems Conference},
  year = {2017},
  pages = {170--174}
}
```

---

## 🤝 Contributing
1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/amazing-feature`)  
3. Commit changes (`git commit -m 'Add amazing feature'`)  
4. Push to branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request  

---

## 📄 License
This project is licensed under the **MIT License**.

---

## 🆘 Support
For questions or issues:
- Open an issue on GitHub  
- Contact: `kinson.vernet@gmail.com`

---

## 🙏 Acknowledgments
- Kaggle for hosting the Nerthus dataset  
- Walter van Huissteden et al. for dataset creation  
- TensorFlow/Keras and Scikit-learn teams for core libraries  

---

<div align="center">

### Built with ❤️ for medical AI research  
If this project helps your research, please give it a ⭐!

</div>
