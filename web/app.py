import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Import our model loader
from loader import modelLoader

# Set page config
st.set_page_config(
    page_title="Nerthus Medical ML",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #4682B4;
        border-bottom: 2px solid #4682B4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-bar {
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #1dd1a1);
        height: 20px;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize models
    if not modelLoader.models_loaded:
        with st.spinner("🔄 Loading AI models... This may take a moment."):
            modelLoader.load_models("XGBoost.joblib", "CNN.keras")
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Nerthus Medical ML</h1>', unsafe_allow_html=True)
    st.markdown("### Automated Bowel Preparation Quality Assessment")
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Section",
        ["🏠 Overview", "📊 Model Performance", "🤖 Live Demo", "📈 Technical Details", "🏥 Medical Context"]
    )
    
    if app_mode == "🏠 Overview":
        show_overview()
    elif app_mode == "📊 Model Performance":
        show_model_performance()
    elif app_mode == "🤖 Live Demo":
        show_live_demo()
    elif app_mode == "📈 Technical Details":
        show_technical_details()
    elif app_mode == "🏥 Medical Context":
        show_medical_context()

def show_live_demo():
    st.markdown('<h2 class="section-header">🤖 Live Prediction Demo</h2>', unsafe_allow_html=True)
    
    # Check model availability
    available_models = modelLoader.get_available_models()
    
    if not any(available_models.values()):
        st.error("""
        ❌ No models loaded. Please ensure:
        - `models/XYZ.joblib` exists
        - Or run the ML pipeline first to train models
        """)
        return
    
    # Show available models
    st.info(f"""
    **Available Models:**
    - 🤖 ML: {'✅' if available_models['ml'] else '❌'}
    - 🧠 CNN: {'✅' if available_models['cnn'] else '❌'}
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📤 Upload Colonoscopy Image")
        
        uploaded_file = st.file_uploader(
            "Choose a colonoscopy image", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a colonoscopy image for BBPS quality assessment"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Model selection based on availability
            model_options = []
            if available_models['ml']:
                model_options.append("ML")
            if available_models['cnn']:
                model_options.append("CNN")
            if len(model_options) > 1:
                model_options.append("Both Models")
            
            if not model_options:
                st.error("No models available for prediction")
                return
                
            model_choice = st.radio(
                "Choose prediction model:",
                model_options,
                horizontal=True
            )
            
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing bowel preparation quality..."):
                    # Small delay for better UX
                    import time
                    time.sleep(1)
                    
                    results = {}
                    errors = {}
                    
                    # CNN Prediction
                    if model_choice in ["CNN", "Both Models"] and available_models['cnn']:
                        cnn_class, cnn_confidence, cnn_error = modelLoader.predict_with_cnn(image)
                        if cnn_class is not None:
                            results['CNN'] = {
                                'class': cnn_class,
                                'confidence': cnn_confidence,
                                'description': modelLoader.get_class_description(cnn_class)
                            }
                        else:
                            errors['CNN'] = cnn_error
                    
                    # ML Prediction
                    if model_choice in ["ML", "Both Models"] and available_models['ml']:
                        rf_class, rf_confidence, rf_error = modelLoader.predict_with_ml(image)
                        if rf_class is not None:
                            results['ML'] = {
                                'class': rf_class,
                                'confidence': rf_confidence,
                                'description': modelLoader.get_class_description(rf_class)
                            }
                        else:
                            errors['ML'] = rf_error
                    
                    # Display results or errors
                    if results:
                        st.success("✅ Analysis Complete!")
                        display_prediction_results(results, model_choice)
                    else:
                        st.error("❌ All predictions failed")
                        
                    # Show any individual model errors
                    for model_name, error in errors.items():
                        st.warning(f"⚠️ {model_name} failed: {error}")
    
    with col2:
        st.markdown("### 🏥 BBPS Scale Reference")
        
        bbps_info = {
            0: "**Score 0: Unprepared** - Mucosa not visible due to solid stool",
            1: "**Score 1: Partially Prepared** - Portions of mucosa visible", 
            2: "**Score 2: Well Prepared** - Minor residue, mucosa clearly visible",
            3: "**Score 3: Excellent Preparation** - Entire mucosa clearly visible"
        }
        
        for score, description in bbps_info.items():
            with st.expander(f"📊 BBPS {score}"):
                st.write(description)
        
        st.markdown("""
        ### 💡 Clinical Notes
        
        - **BBPS** = Boston Bowel Preparation Scale
        - Used worldwide for standardized assessment
        - Automated scoring reduces variability between clinicians
        - Quality affects adenoma detection rates by 20-50%
        
        ### 🎯 Model Performance
        - **CNN**: 92.4% validation accuracy
        - **ML**: 90.5% cross-validation accuracy
        - Both models trained on 5,525 medical images
        """)

def display_prediction_results(results, model_choice):
    """Display prediction results in a nice format"""
    
    st.markdown("### 📋 BBPS Assessment Results")
    
    if model_choice == "Both Models":
        # Compare both models
        col1, col2 = st.columns(2)
        
        with col1:
            if 'CNN' in results:
                display_single_prediction(results['CNN'], "CNN Champion", "#FFD700")
            else:
                st.warning("⚠️ CNN prediction not available")
        
        with col2:
            if 'ML' in results:
                display_single_prediction(results['ML'], "ML", "#2E8B57")
            else:
                st.warning("⚠️ ML prediction not available")
        
        # Show agreement if both models succeeded
        if 'CNN' in results and 'ML' in results:
            cnn_class = results['CNN']['class']
            rf_class = results['ML']['class']
            
            if cnn_class == rf_class:
                st.success(f"✅ Models agree: BBPS Score {cnn_class}")
            else:
                st.warning(f"⚠️ Models disagree: CNN={cnn_class}, RF={rf_class}")
                
    else:
        # Single model result - handle different naming cases
        if "CNN" in model_choice and 'CNN' in results:
            result = results['CNN']
            model_name = "CNN Champion"
            color = "#FFD700"
        elif "ML" in model_choice and 'ML' in results:
            result = results['ML']
            model_name = "ML"
            color = "#2E8B57"
        else:
            # Fallback: use the first available result
            available_models = list(results.keys())
            if available_models:
                model_name = available_models[0]
                result = results[model_name]
                color = "#4682B4"
            else:
                st.error("❌ No prediction results available")
                return
        
        display_single_prediction(result, model_name, color)

def display_single_prediction(result, model_name, color):
    """Display a single model prediction"""
    
    # Create confidence visual
    confidence_percent = result['confidence'] * 100
    
    st.markdown(f'<div class="prediction-card">', unsafe_allow_html=True)
    
    st.markdown(f"#### 🎯 {model_name}")
    st.markdown(f"**Predicted BBPS Score:** `{result['class']}`")
    st.markdown(f"**Confidence:** `{confidence_percent:.1f}%`")
    
    # Confidence bar
    st.markdown(f"""
    <div style="width: 100%; background: #e0e0e0; border-radius: 10px; margin: 10px 0;">
        <div style="width: {confidence_percent}%; background: {color}; height: 20px; border-radius: 10px; 
                    display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
            {confidence_percent:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**Description:** {result['description']}")
    
    # Confidence interpretation
    if result['confidence'] > 0.9:
        st.markdown("🟢 **High confidence** - Very reliable prediction")
    elif result['confidence'] > 0.7:
        st.markdown("🟡 **Good confidence** - Reliable prediction") 
    else:
        st.markdown("🔴 **Low confidence** - Consider re-analyzing")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_overview():
    st.markdown('<h2 class="section-header">Project Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Project Summary
        
        This web application demonstrates a state-of-the-art medical AI system for 
        **automated bowel preparation quality assessment** using colonoscopy images.
        
        **Key Features:**
        - 🏥 **Medical Grade**: 92.4% accurate BBPS scoring
        - 🤖 **Dual Approach**: Both traditional ML and Deep Learning
        - ⚡ **Real-time**: Instant predictions on new images
        - 📊 **Comprehensive**: Full model analysis and comparison
        
        **Clinical Impact:**
        - Reduces inter-observer variability in colonoscopy quality assessment
        - Enables standardized bowel preparation scoring
        - Supports clinical decision making
        """)
    
    with col2:
        # Placeholder for project architecture image
        st.info("""
        **Project Architecture:**
        - Traditional ML: ML, XGBoost
        - Deep Learning: Custom CNN
        - 22 Medical Image Features
        - Production-ready Pipeline
        """)
    
    # Key metrics
    st.markdown("### 🏆 Key Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("CNN Accuracy", "92.4%", "+1.9% vs RF")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("ML", "90.5%", "Traditional ML")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Training Images", "5,525", "Medical Dataset")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("BBPS Classes", "4", "0-3 Scale")
        st.markdown('</div>', unsafe_allow_html=True)

def show_model_performance():
    st.markdown('<h2 class="section-header">Model Performance Comparison</h2>', unsafe_allow_html=True)
    
    # Model comparison data - USING YOUR ACTUAL RESULTS
    models_data = {
        'Model': ['CNN Champion', 'ML', 'XGBoost', 'Logistic Regression', 'SVM'],
        'Accuracy': [0.924, 0.905, 0.897, 0.805, 0.625],
        'Type': ['Deep Learning', 'Traditional ML', 'Traditional ML', 'Traditional ML', 'Traditional ML'],
        'Training Time (min)': [61, 3, 4, 1, 2]
    }
    df = pd.DataFrame(models_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Interactive bar chart
        fig = px.bar(df, x='Model', y='Accuracy', color='Type',
                    title='Model Accuracy Comparison (Your Actual Results)',
                    color_discrete_map={'Deep Learning': '#FFD700', 'Traditional ML': '#2E8B57'})
        fig.update_layout(yaxis_tickformat='.1%', yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Performance metrics
        st.markdown("### 📊 Detailed Performance")
        
        # Create a nice table
        fig = go.Figure(data=[go.Table(
            header=dict(values=['Model', 'Accuracy', 'Training Time'],
                        fill_color='#2E8B57',
                        align='left',
                        font=dict(color='white', size=12)),
            cells=dict(values=[df.Model, 
                             [f'{acc:.1%}' for acc in df.Accuracy],
                             [f'{time} min' for time in df['Training Time (min)']],
                            ],
                      align='left'))
        ])
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance - USING YOUR ACTUAL TOP FEATURES
    st.markdown("### 🔍 Feature Importance Analysis")
    
    features_data = {
        'Feature': ['min_intensity', 'hue_mean', 'homogeneity', 'edge_density', 'saturation_mean',
                   'contrast', 'energy', 'correlation', 'sharpness', 'image_entropy'],
        'Importance': [0.125, 0.098, 0.087, 0.076, 0.065, 0.054, 0.048, 0.042, 0.037, 0.032]
    }
    features_df = pd.DataFrame(features_data)
    
    fig = px.bar(features_df.head(8), x='Importance', y='Feature', 
                 orientation='h', title='Top 8 Most Important Features (From Your Analysis)',
                 color='Importance', color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)

def show_technical_details():
    st.markdown('<h2 class="section-header">📈 Technical Implementation</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🧠 ML Pipeline", "🕸️ CNN Architecture", "⚙️ Feature Engineering"])
    
    with tab1:
        st.markdown("""
        ### Traditional Machine Learning Pipeline
        
        **Algorithms Implemented:**
        - ML (90.5% accuracy)
        - XGBoost (89.7% accuracy) 
        - Logistic Regression (80.5% accuracy)
        - SVM (62.5% accuracy)
        
        **Validation Strategy:**
        - 5-fold stratified cross-validation
        - Train-test split with class balancing
        - Overfitting detection with performance gaps
        
        **Key Features:**
        ```python
        # Example feature extraction
        features = {
            'texture': ['contrast', 'homogeneity', 'energy', 'correlation'],
            'color': ['hue_mean', 'saturation_mean', 'l_mean', 'a_mean', 'b_mean'],
            'edges': ['edge_density', 'sharpness'],
            'intensity': ['mean_intensity', 'std_intensity', 'min_intensity']
        }
        ```
        """)
    
    with tab2:
        st.markdown("""
        ### Deep Learning CNN Architecture
        
        **Model Architecture:**
        ```python
        Model: "sequential"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        conv2d (Conv2D)              (None, 148, 148, 32)      896       
        max_pooling2d (MaxPooling2D) (None, 74, 74, 32)        0         
        dropout (Dropout)            (None, 74, 74, 32)        0         
        conv2d_1 (Conv2D)            (None, 72, 72, 64)        18496     
        max_pooling2d_1 (MaxPooling2 (None, 36, 36, 64)        0         
        dropout_1 (Dropout)          (None, 36, 36, 64)        0         
        conv2d_2 (Conv2D)            (None, 34, 34, 128)       73856     
        max_pooling2d_2 (MaxPooling2 (None, 17, 17, 128)       0         
        global_average_pooling2d (Gl (None, 128)               0         
        dropout_2 (Dropout)          (None, 128)               0         
        dense (Dense)                (None, 4)                 516       
        =================================================================
        Total params: 93,764
        ```
        
        **Training Configuration:**
        - Optimizer: Adam (lr=0.001)
        - Loss: Sparse Categorical Crossentropy  
        - Callbacks: Early stopping, LR reduction
        - Regularization: Dropout, batch normalization
        """)
    
    with tab3:
        st.markdown("""
        ### Medical Image Feature Engineering
        
        **22 Handcrafted Features:**
        
        **Texture Analysis (GLCM):**
        - Contrast, Homogeneity, Energy, Correlation
        
        **Color Space Analysis:**
        - RGB: Mean intensities
        - HSV: Hue, Saturation, Value means  
        - LAB: L*, a*, b* means
        
        **Edge and Shape Analysis:**
        - Edge density, Sharpness (Laplacian variance)
        
        **Intensity Statistics:**
        - Mean, Standard deviation, Min, Max, Median
        
        **Advanced Features:**
        - Image entropy, LBP entropy, Blob count
        
        **Top 5 Most Predictive Features:**
        1. `min_intensity` - Minimum image intensity
        2. `hue_mean` - Average hue in HSV space  
        3. `homogeneity` - Texture homogeneity
        4. `edge_density` - Edge concentration
        5. `saturation_mean` - Color saturation
        """)

def show_medical_context():
    st.markdown('<h2 class="section-header">🏥 Medical Background</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🩺 Clinical Significance
        
        **Bowel Preparation Quality** is critical for:
        - **Colonoscopy effectiveness**
        - **Adenoma detection rates** (cancer precursors)
        - **Patient safety and comfort**
        - **Healthcare resource optimization**
        
        **Impact on Cancer Detection:**
        - Poor preparation → 20-50% lower adenoma detection
        - Inadequate cleansing → missed lesions
        - Quality affects screening interval decisions
        
        **Current Challenges:**
        - Inter-observer variability in scoring
        - Subjective assessment
        - Lack of standardization
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Boston Bowel Preparation Scale (BBPS)
        
        **Standardized Scoring System:**
        
        **Score 0: Unprepared**
        - Mucosa not visible due to solid stool
        - Cannot be interpreted
        
        **Score 1: Partially Prepared**  
        - Portions of mucosa visible
        - Other areas not well seen due to staining, debris
        
        **Score 2: Well Prepared**
        - Minor residue of staining, debris
        - Mucosa clearly visible
        
        **Score 3: Excellent Preparation**
        - Entire mucosa clearly visible
        - No residual staining, small debris
        """)
    
    st.markdown("""
    ### 🎯 Project Clinical Impact
    
    **Automated BBPS scoring provides:**
    - ✅ Standardized, consistent assessments
    - ✅ Reduced inter-observer variability  
    - ✅ Real-time quality feedback during procedures
    - ✅ Objective quality metrics for research
    - ✅ Training tool for new gastroenterologists
    
    **Potential Clinical Applications:**
    - Real-time quality monitoring during colonoscopy
    - Automated procedure reporting
    - Quality assurance programs
    - Research and clinical trials
    """)

if __name__ == "__main__":
    main()