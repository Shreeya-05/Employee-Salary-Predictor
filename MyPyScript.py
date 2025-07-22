import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# Load trained models and transformers (with error handling)
@st.cache_resource
def load_models():
    try:
        regressor = joblib.load("best_regressor.pkl")
        classifier = joblib.load("best_classifier.pkl")
        scaler = joblib.load("scaler.pkl")
        encoders = joblib.load("encoders.pkl")
        return regressor, classifier, scaler, encoders
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Using mock predictions for demo.")
        return None, None, None, None

# Mock prediction functions
def mock_predict_salary(age, gender, education, job_title, experience):
    base_salary = 40000
    base_salary += (age - 25) * 800
    base_salary += experience * 3200
    
    education_multiplier = {
        "High School": 1.0,
        "Associate's": 1.15,
        "Bachelor's": 1.35,
        "Master's": 1.65,
        "PhD": 2.0
    }.get(education, 1.0)
    
    job_multiplier = {
        "Software Engineer": 1.4,
        "Data Scientist": 1.6,
        "Data Analyst": 1.2,
        "HR": 1.0,
        "Manager": 1.3,
        "Sales Executive": 1.1
    }.get(job_title, 1.0)
    
    base_salary *= education_multiplier * job_multiplier
    
    if gender == "Female":
        base_salary *= 0.95
    
    base_salary += (np.random.random() - 0.5) * 10000
    return max(30000, int(base_salary))

def mock_predict_level(salary):
    if salary < 60000:
        return "Low", np.random.uniform(0.85, 0.95)
    elif salary < 100000:
        return "Medium", np.random.uniform(0.80, 0.95)
    else:
        return "High", np.random.uniform(0.88, 0.98)

# Page configuration
st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS matching the Next.js version exactly
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    .main > div {
        padding: 0;
        max-width: none;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f0f9ff 0%, #ffffff 50%, #faf5ff 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .header-container {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(229, 231, 235, 0.8);
        padding: 1.5rem 0;
        position: sticky;
        top: 0;
        z-index: 100;
        margin-bottom: 2rem;
    }
    
    .header-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .header-icon {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        padding: 1.0rem;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem !important;
    }
    
    .header-text h1 {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        font-size: 4rem !important;
        font-weight: 700;
        line-height: 1.2;
    }
    
    .header-text p {
        color: #6b7280;
        margin: 0;
        font-size: 0.875rem;
        font-weight: 400;
    }
    
    /* Main content container */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem 3rem 2rem;
    }
    
    /* Card styling */
    .card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        border: 0;
        border-radius: 16px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        padding: 1.5rem;
        height: fit-content;
    }
    
    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .section-icon {
        color: #2563eb;
        font-size: 1.25rem;
    }
    
    .section-title {
        color: #1f2937;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 0;
    }
    
    .section-subtitle {
        color: #6b7280;
        font-size: 0.875rem;
        margin-bottom: 1.5rem;
        line-height: 1.4;
    }
    
    /* Form field styling */
    .form-field {
        margin-bottom: 1.5rem;
    }
    
    .field-label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
        color: #374151;
        font-size: 0.875rem;
    }
    
    .field-icon {
        color: #6b7280;
        font-size: 1rem;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
        height: 8px !important;
        border-radius: 4px !important;
    }
    
    .stSlider > div > div > div > div > div {
        background: white !important;
        border: 3px solid #2563eb !important;
        width: 24px !important;
        height: 24px !important;
        border-radius: 50% !important;
        box-shadow: 0 4px 8px rgba(37, 99, 235, 0.3) !important;
        transition : all 0.2s ease !important;
    }
    
    .stSlider > div > div > div > div > div:hover {
        box-shadow: 0 0 0 12px rgba(37, 99, 235, 0.15) !important;
        transform: scale(1.1) !important;
    }
            
    .stSlider > div > div > div > div > div:active {
        transform: scale(0.95) !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
        background: white !important;
        font-size: 0.875rem !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }
    
    /* Tab styling */
    .tab-container {
        display: flex;
        background: #f3f4f6;
        border-radius: 8px;
        padding: 0.25rem;
        margin-bottom: 1.5rem;
        gap: 0.25rem;
    }
    
    .tab-button {
        flex: 1;
        padding: 0.75rem 1rem;
        border: none;
        background: transparent;
        border-radius: 6px;
        cursor: pointer;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        color: #6b7280;
    }
    
    .tab-button.active {
        background: white;
        color: #2563eb;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Prediction card styling */
    .prediction-display {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .prediction-icon {
        font-size: 3rem;
        color: #059669;
        margin-bottom: 1rem;
        display: block;
    }
    
    .prediction-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #065f46;
        margin-bottom: 0.5rem;
    }
    
    .prediction-subtitle {
        color: #047857;
        font-size: 0.875rem;
        line-height: 1.4;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.75rem 1.5rem !important;
        width: 100% !important;
        height: 3rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #6d28d9 100%) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Result display */
    .result-display {
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    
    .result-amount {
        font-size: 2rem;
        font-weight: 700;
        color: #059669;
        margin-bottom: 0.5rem;
    }
    
    .result-label {
        color: #6b7280;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .level-badge {
        background: #2563eb;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        display: inline-block;
    }
    
    .level-badge.high {
        background: #059669;
    }
    
    .level-badge.medium {
        background: #d97706;
    }
    
    .level-badge.low {
        background: #dc2626;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.875rem;
        margin-top: 3rem;
        padding: 1rem;
    }
    
    /* Hide Streamlit elements */
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    .stMainBlockContainer {padding-top: 0;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-container {
            padding: 0 1rem 2rem 1rem;
        }
        
        .header-content {
            padding: 0 1rem;
        }
        
        .card {
            margin-bottom: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_mode' not in st.session_state:
    st.session_state.prediction_mode = 'salary'

# Header
st.markdown("""
<div class="header-container">
    <div class="header-content">
        <div class="header-icon">
            💲
        </div>
        <div class="header-text">
            <h1>AI Salary Predictor</h1>
            <p>Predict employee compensation with machine learning</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Load models
regressor, classifier, scaler, encoders = load_models()

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Create two columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
    <div class="card">
        <div class="section-header">
            <span class="section-icon">💼</span>
            <h2 class="section-title">Employee Details</h2>
        </div>
        <p class="section-subtitle">Enter the employee information to get salary predictions</p>
    """, unsafe_allow_html=True)
    
    # Age field
    st.markdown('<div class="form-field">', unsafe_allow_html=True)
    st.markdown('<div class="field-label"><span class="field-icon">📅</span>Age: 30 years</div>', unsafe_allow_html=True)
    # Custom blue slider style for Age
    st.markdown("""
    <style>
    /* Blue slider track and thumb for Age slider */
    div[data-testid="stSlider"][aria-label="age_slider"] .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%) !important;
    }
    div[data-testid="stSlider"][aria-label="age_slider"] .stSlider > div > div > div > div > div {
        border: 3px solid #1e3a8a !important;
    }
    </style>
    """, unsafe_allow_html=True)
    age = st.slider("", min_value=18, max_value=65, value=30, key="age_slider", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Gender field
    st.markdown('<div class="form-field">', unsafe_allow_html=True)
    st.markdown('<div class="field-label"><span class="field-icon">👥</span>Gender</div>', unsafe_allow_html=True)
    gender = st.selectbox("", ["Select gender", "Male", "Female"], key="gender_select", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Education field
    st.markdown('<div class="form-field">', unsafe_allow_html=True)
    st.markdown('<div class="field-label"><span class="field-icon">🎓</span>Education Level</div>', unsafe_allow_html=True)
    education = st.selectbox("", ["Select education level", "High School", "Associate's", "Bachelor's", "Master's", "PhD"], key="education_select", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Job Title field
    st.markdown('<div class="form-field">', unsafe_allow_html=True)
    st.markdown('<div class="field-label"><span class="field-icon">💼</span>Job Title</div>', unsafe_allow_html=True)
    job_title = st.selectbox("", ["Select job title", "Software Engineer", "Data Scientist", "Data Analyst", "HR", "Manager", "Sales Executive"], key="job_select", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Experience field
    st.markdown('<div class="form-field">', unsafe_allow_html=True)
    st.markdown(f'<div class="field-label"><span class="field-icon">📈</span>Years of Experience: {st.session_state.get("experience_slider", 5)} years</div>', unsafe_allow_html=True)
    experience = st.slider("", min_value=0, max_value=40, value=5, key="experience_slider", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close card

with col2:
    st.markdown("""
    <div class="card">
        <div class="section-header">
            <span class="section-icon">📊</span>
            <h2 class="section-title">Prediction Results</h2>
        </div>
        <p class="section-subtitle">Choose your prediction type and get AI-powered insights</p>
    """, unsafe_allow_html=True)
    
    # Tab buttons
    col_tab1, col_tab2 = st.columns(2)
    
    with col_tab1:
        if st.button("💲 Salary Amount", key="salary_tab", use_container_width=True):
            st.session_state.prediction_mode = 'salary'
    
    with col_tab2:
        if st.button("🏷️ Salary Level", key="level_tab", use_container_width=True):
            st.session_state.prediction_mode = 'level'
    
    # Prediction display area
    if st.session_state.prediction_mode == 'salary':
        st.markdown("""
        <div class="prediction-display">
            <span class="prediction-icon">💲</span>
            <div class="prediction-title">Salary Prediction</div>
            <div class="prediction-subtitle">Get the exact predicted salary amount</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="prediction-display" style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-color: #3b82f6;">
            <span class="prediction-icon" style="color: #1d4ed8;">🏷️</span>
            <div class="prediction-title" style="color: #1e3a8a;">Salary Level</div>
            <div class="prediction-subtitle" style="color: #1e40af;">Classify salary as Low, Medium, or High</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Predict button
    predict_clicked = st.button("Predict Salary", key="predict_btn", use_container_width=True, type="primary")
    
    # Handle prediction results
    if predict_clicked:
        if gender == "Select gender" or education == "Select education level" or job_title == "Select job title":
            st.error("⚠️ Please fill in all required fields!")
        else:
            with st.spinner("🤖 AI is analyzing the data..."):
                import time
                time.sleep(1.5)
                
                try:
                    if regressor and classifier and scaler and encoders:
                        # Real model prediction
                        input_dict = {
                            "Age": age,
                            "Gender": gender,
                            "Education Level": education,
                            "Job Title": job_title,
                            "Years of Experience": experience
                        }
                        
                        input_df = pd.DataFrame([input_dict])
                        input_df_encoded = input_df.copy()
                        
                        # Encode categorical columns
                        for col in input_df_encoded.select_dtypes(include="object").columns:
                            if col in encoders:
                                input_df_encoded[col] = encoders[col].transform(input_df_encoded[col])
                        
                        # Scale features
                        input_scaled = scaler.transform(input_df_encoded)
                        
                        if st.session_state.prediction_mode == 'salary':
                            salary_pred = regressor.predict(input_scaled)[0]
                            st.markdown(f"""
                            <div class="result-display">
                                <div class="result-amount">${salary_pred:,.0f}</div>
                                <div class="result-label">Predicted Annual Salary</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            class_pred = classifier.predict(input_scaled)[0]
                            class_proba = classifier.predict_proba(input_scaled)[0]
                            confidence = max(class_proba) * 100
                            
                            label_map = {0: "Low", 1: "Medium", 2: "High"}
                            predicted_level = label_map.get(class_pred, 'Unknown')
                            badge_class = predicted_level.lower()
                            
                            st.markdown(f"""
                            <div class="result-display">
                                <div class="level-badge {badge_class}">{predicted_level} Salary Level</div><br>
                                <div class="result-label">Confidence: {confidence:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    else:
                        # Mock predictions
                        if st.session_state.prediction_mode == 'salary':
                            salary_pred = mock_predict_salary(age, gender, education, job_title, experience)
                            st.markdown(f"""
                            <div class="result-display">
                                <div class="result-amount">${salary_pred:,}</div>
                                <div class="result-label">Predicted Annual Salary</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            mock_salary = mock_predict_salary(age, gender, education, job_title, experience)
                            predicted_level, confidence = mock_predict_level(mock_salary)
                            badge_class = predicted_level.lower()
                            
                            st.markdown(f"""
                            <div class="result-display">
                                <div class="level-badge {badge_class}">{predicted_level} Salary Level</div><br>
                                <div class="result-label">Confidence: {confidence*100:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close card

# Footer
st.markdown("""
<div class="footer">
    <p>Powered by advanced machine learning algorithms • Built with Streamlit & Python</p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close main container