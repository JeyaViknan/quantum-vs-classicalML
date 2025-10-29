import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
import warnings
warnings.filterwarnings('ignore')

# This avoids the infinite loop issue while maintaining quantum principles
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_algorithms.utils import algorithm_globals
    QISKIT_AVAILABLE = True
    QISKIT_ERROR = None
except ImportError as e:
    QISKIT_AVAILABLE = False
    QISKIT_ERROR = str(e)

# Set page config
st.set_page_config(
    page_title="Crop Yield ML Comparison",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Dark mode background */
    .stApp {
        background-color: #0e1117;
        color: #e6edf3;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #58a6ff;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #8b949e;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #161b22;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #58a6ff;
        color: #e6edf3;
    }
    
    /* Dark mode for metric boxes */
    .stMetric {
        background-color: #161b22 !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
        border: 1px solid #30363d !important;
    }
    
    .stMetric label {
        color: #8b949e !important;
        font-size: 0.9rem !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-size: 1.8rem !important;
        font-weight: bold !important;
    }
    
    /* Dark mode for dataframe */
    .stDataFrame {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
    }
    
    /* Dark mode for tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0d1117;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #8b949e;
        background-color: #161b22;
    }
    
    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        background-color: #0d1117 !important;
    }
    
    /* Dark mode for buttons */
    .stButton > button {
        background-color: #238636 !important;
        color: #ffffff !important;
        border: 1px solid #2ea043 !important;
    }
    
    .stButton > button:hover {
        background-color: #2ea043 !important;
    }
    
    /* Dark mode for expander */
    .streamlit-expanderHeader {
        background-color: #161b22 !important;
        color: #58a6ff !important;
    }
    
    /* Dark mode for sidebar */
    .stSidebar {
        background-color: #0d1117 !important;
    }
    
    .stSidebar [data-testid="stMarkdownContainer"] {
        color: #e6edf3 !important;
    }
    
    /* Dark mode for selectbox and inputs */
    .stSelectbox, .stSlider, .stNumberInput {
        color: #e6edf3 !important;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background-color: #161b22 !important;
    }
    
    /* Dark mode for info/warning boxes */
    .stInfo {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
    }
    
    .stWarning {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
    }
    
    .stError {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
    }
    
    /* Dark mode for markdown */
    .stMarkdown {
        color: #e6edf3 !important;
    }
    
    /* Dark mode for progress bar */
    .stProgress > div > div > div {
        background-color: #238636 !important;
    }
    
    /* Dark mode for spinner */
    .stSpinner {
        color: #58a6ff !important;
    }
    
    /* Ensure text is always visible */
    h1, h2, h3, h4, h5, h6 {
        color: #e6edf3 !important;
    }
    
    p, span, div {
        color: #e6edf3 !important;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🌾 Crop Yield Prediction: Classical vs Quantum ML</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Compare Random Forest, SVR, and Quantum-Inspired SVR models for agricultural yield forecasting</div>', unsafe_allow_html=True)

class QuantumInspiredKernel:
    """
    Quantum-inspired kernel using periodic feature encoding.
    Avoids expensive quantum circuit evaluations while maintaining quantum principles.
    """
    def __init__(self, n_qubits=4):
        self.n_qubits = n_qubits
    
    def _encode_features(self, X):
        """Encode features using periodic quantum-inspired encoding"""
        # Use sine and cosine transformations to create quantum-like feature space
        encoded = []
        for i in range(self.n_qubits):
            angle = (i + 1) * np.pi / self.n_qubits
            encoded.append(np.sin(angle * X))
            encoded.append(np.cos(angle * X))
        return np.column_stack(encoded)
    
    def __call__(self, X1, X2):
        """Compute kernel matrix between X1 and X2"""
        X1_encoded = self._encode_features(X1)
        X2_encoded = self._encode_features(X2)
        
        # Compute RBF kernel on encoded features
        gamma = 1.0 / X1_encoded.shape[1]
        kernel_matrix = np.exp(-gamma * np.sum((X1_encoded[:, np.newaxis, :] - X2_encoded[np.newaxis, :, :]) ** 2, axis=2))
        return kernel_matrix

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# Data loading with caching
@st.cache_data
def load_data():
    """Load and validate the crop yield dataset"""
    try:
        df = pd.read_csv('yield_df.csv', index_col=0)
        
        # Validate required columns
        required_cols = ['Area', 'Item', 'Year', 'hg/ha_yield', 
                        'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
        
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing required columns. Expected: {required_cols}")
            return None
        
        # Remove any rows with missing values
        df = df.dropna()
        
        return df
    except FileNotFoundError:
        st.error("❌ Error: yield_df.csv not found. Please ensure the data file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

@st.cache_data
def analyze_dataset(df):
    """Perform comprehensive dataset analysis"""
    analysis = {
        'total_samples': len(df),
        'crops': df['Item'].unique(),
        'countries': df['Area'].unique(),
        'year_range': (df['Year'].min(), df['Year'].max()),
        'yield_stats': {
            'mean': df['hg/ha_yield'].mean(),
            'std': df['hg/ha_yield'].std(),
            'min': df['hg/ha_yield'].min(),
            'max': df['hg/ha_yield'].max()
        },
        'rainfall_stats': {
            'mean': df['average_rain_fall_mm_per_year'].mean(),
            'std': df['average_rain_fall_mm_per_year'].std()
        },
        'pesticides_stats': {
            'mean': df['pesticides_tonnes'].mean(),
            'std': df['pesticides_tonnes'].std()
        },
        'temp_stats': {
            'mean': df['avg_temp'].mean(),
            'std': df['avg_temp'].std()
        }
    }
    return analysis

# Load data
df = load_data()

if df is not None:
    # Tabs for navigation
    tab1, tab2, tab3 = st.tabs(["🔬 Model Comparison", "📊 Dataset Analysis", "ℹ️ About"])
    
    with tab1:
        # Crop selection
        crops = sorted(df['Item'].unique())
        selected_crop = st.sidebar.selectbox("🌱 Select Crop", crops, index=0)
        
        # Model selection
        st.sidebar.markdown("### 🤖 Model Selection")
        enable_rf = st.sidebar.checkbox("Random Forest", value=True)
        enable_svr = st.sidebar.checkbox("Support Vector Regression", value=True)
        enable_quantum = st.sidebar.checkbox("⚛️ Quantum-Inspired SVR", value=True)
        
        # Advanced settings
        with st.sidebar.expander("🔧 Advanced Settings"):
            test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
            random_state = st.number_input("Random State", 0, 100, 42)
            n_estimators = st.slider("RF: Number of Trees", 50, 500, 100, 50)
            max_depth = st.slider("RF: Max Depth", 5, 30, 10)
            svr_c = st.slider("SVR: C Parameter", 0.1, 10.0, 1.0, 0.1)
            svr_epsilon = st.slider("SVR: Epsilon", 0.01, 1.0, 0.1, 0.01)
        
        st.sidebar.markdown("---")
        
        # Run button
        run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        # Filter data for selected crop
        crop_df = df[df['Item'] == selected_crop].copy()
        
        # Display dataset info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Samples", len(crop_df))
        with col2:
            st.metric("🌍 Countries", crop_df['Area'].nunique())
        with col3:
            st.metric("📅 Year Range", f"{crop_df['Year'].min()}-{crop_df['Year'].max()}")
        with col4:
            st.metric("🌾 Selected Crop", selected_crop)
        
        st.markdown("---")
        
        if run_analysis:
            if not (enable_rf or enable_svr or enable_quantum):
                st.warning("⚠️ Please select at least one model to train.")
            else:
                # Prepare data
                with st.spinner("📊 Preparing data..."):
                    X = crop_df[['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']]
                    y = crop_df['hg/ha_yield']
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    y_scaler = StandardScaler()
                    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
                    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

                # Store results
                results = {}
                
                # Train models
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                models_to_train = []
                if enable_rf:
                    models_to_train.append('Random Forest')
                if enable_svr:
                    models_to_train.append('SVR')
                if enable_quantum:
                    models_to_train.append('Quantum-Inspired SVR')
                
                total_models = len(models_to_train)
                
                for idx, model_name in enumerate(models_to_train):
                    status_text.text(f"🔄 Training {model_name}... ({idx+1}/{total_models})")
                    
                    if model_name == 'Random Forest':
                        # Random Forest
                        start_time = time.time()
                        rf_model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=random_state,
                            n_jobs=-1
                        )
                        rf_model.fit(X_train_scaled, y_train)
                        y_pred_rf = rf_model.predict(X_test_scaled)
                        train_time_rf = time.time() - start_time
                        
                        # Cross-validation
                        cv_scores_rf = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='r2')
                        
                        results['Random Forest'] = {
                            'model': rf_model,
                            'predictions': y_pred_rf,
                            'r2': r2_score(y_test, y_pred_rf),
                            'mae': mean_absolute_error(y_test, y_pred_rf),
                            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
                            'train_time': train_time_rf,
                            'cv_mean': cv_scores_rf.mean(),
                            'cv_std': cv_scores_rf.std(),
                            'feature_importance': rf_model.feature_importances_
                        }
                    
                    elif model_name == 'SVR':
                        # Support Vector Regression
                        start_time = time.time()
                        svr_model = SVR(C=svr_c, epsilon=svr_epsilon, kernel='rbf', gamma='scale')
                        svr_model.fit(X_train_scaled, y_train_scaled)
                        y_pred_svr_scaled = svr_model.predict(X_test_scaled)
                        y_pred_svr = y_scaler.inverse_transform(y_pred_svr_scaled.reshape(-1, 1)).ravel()
                        train_time_svr = time.time() - start_time
                        
                        # Cross-validation
                        cv_scores_svr = cross_val_score(svr_model, X_train_scaled, y_train_scaled, cv=5, scoring='r2')
                        
                        results['SVR'] = {
                            'model': svr_model,
                            'predictions': y_pred_svr,
                            'r2': r2_score(y_test, y_pred_svr),
                            'mae': mean_absolute_error(y_test, y_pred_svr),
                            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_svr)),
                            'train_time': train_time_svr,
                            'cv_mean': cv_scores_svr.mean(),
                            'cv_std': cv_scores_svr.std()
                        }
                    
                    elif model_name == 'Quantum-Inspired SVR':
                        try:
                            start_time = time.time()
                            
                            # Create quantum-inspired kernel
                            qi_kernel = QuantumInspiredKernel(n_qubits=4)
                            
                            qi_svr_model = SVR(C=svr_c, epsilon=svr_epsilon, kernel=qi_kernel)
                            qi_svr_model.fit(X_train_scaled, y_train_scaled)
                            y_pred_qi_svr_scaled = qi_svr_model.predict(X_test_scaled)
                            y_pred_qi_svr = y_scaler.inverse_transform(y_pred_qi_svr_scaled.reshape(-1, 1)).ravel()
                            train_time_qi_svr = time.time() - start_time
                            
                            # Cross-validation
                            cv_scores_qi_svr = cross_val_score(qi_svr_model, X_train_scaled, y_train_scaled, cv=5, scoring='r2')
                            
                            results['Quantum-Inspired SVR'] = {
                                'model': qi_svr_model,
                                'predictions': y_pred_qi_svr,
                                'r2': r2_score(y_test, y_pred_qi_svr),
                                'mae': mean_absolute_error(y_test, y_pred_qi_svr),
                                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_qi_svr)),
                                'train_time': train_time_qi_svr,
                                'cv_mean': cv_scores_qi_svr.mean(),
                                'cv_std': cv_scores_qi_svr.std()
                            }
                        except Exception as e:
                            st.error(f"❌ Quantum-Inspired model error: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / total_models)
                
                status_text.text("✅ Training complete!")
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                st.markdown("## 📊 Model Performance Comparison")
                
                # Create comparison table
                comparison_data = []
                for model_name, result in results.items():
                    comparison_data.append({
                        'Model': model_name,
                        'R² Score': f"{result['r2']:.4f}",
                        'MAE': f"{result['mae']:.2f}",
                        'RMSE': f"{result['rmse']:.2f}",
                        'Training Time (s)': f"{result['train_time']:.3f}",
                        'CV R² (mean±std)': f"{result['cv_mean']:.4f}±{result['cv_std']:.4f}"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Metrics in columns
                st.markdown("### 📈 Detailed Metrics")
                cols = st.columns(len(results))
                
                for idx, (model_name, result) in enumerate(results.items()):
                    with cols[idx]:
                        st.markdown(f"**{model_name}**")
                        st.metric("R² Score", f"{result['r2']:.4f}")
                        st.metric("MAE", f"{result['mae']:.2f}")
                        st.metric("RMSE", f"{result['rmse']:.2f}")
                        st.metric("Training Time", f"{result['train_time']:.3f}s")
                
                st.markdown("---")
                
                # Visualizations
                st.markdown("## 📉 Visualizations")
                
                # Actual vs Predicted plots
                st.markdown("### Actual vs Predicted Values")
                
                num_models = len(results)
                cols = st.columns(min(num_models, 3))
                
                for idx, (model_name, result) in enumerate(results.items()):
                    with cols[idx % 3]:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        
                        # Scatter plot
                        ax.scatter(y_test, result['predictions'], alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
                        
                        # Perfect prediction line
                        min_val = min(y_test.min(), result['predictions'].min())
                        max_val = max(y_test.max(), result['predictions'].max())
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
                        
                        ax.set_xlabel('Actual Yield (hg/ha)', fontsize=11, fontweight='bold')
                        ax.set_ylabel('Predicted Yield (hg/ha)', fontsize=11, fontweight='bold')
                        ax.set_title(f'{model_name}\nR² = {result["r2"]:.4f}', fontsize=12, fontweight='bold')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        plt.close()
                
                # Error distribution
                st.markdown("### Error Distribution")
                
                fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 5))
                if num_models == 1:
                    axes = [axes]
                
                for idx, (model_name, result) in enumerate(results.items()):
                    errors = y_test - result['predictions']
                    
                    axes[idx].hist(errors, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
                    axes[idx].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
                    axes[idx].set_xlabel('Prediction Error (hg/ha)', fontsize=11, fontweight='bold')
                    axes[idx].set_ylabel('Frequency', fontsize=11, fontweight='bold')
                    axes[idx].set_title(f'{model_name}\nMean Error: {errors.mean():.2f}', fontsize=12, fontweight='bold')
                    axes[idx].legend()
                    axes[idx].grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Feature importance for Random Forest
                if 'Random Forest' in results:
                    st.markdown("### 🌟 Feature Importance (Random Forest)")
                    
                    feature_names = ['Year', 'Rainfall (mm/year)', 'Pesticides (tonnes)', 'Avg Temperature (°C)']
                    importance = results['Random Forest']['feature_importance']
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
                    bars = ax.barh(feature_names, importance, color=colors, edgecolor='black', linewidth=1.5)
                    
                    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
                    ax.set_title('Feature Importance in Random Forest Model', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    # Add value labels
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width, bar.get_y() + bar.get_height()/2, 
                               f'{width:.3f}', ha='left', va='center', fontweight='bold', fontsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                # Residual plots
                st.markdown("### 📊 Residual Analysis")
                
                cols = st.columns(min(num_models, 3))
                
                for idx, (model_name, result) in enumerate(results.items()):
                    with cols[idx % 3]:
                        residuals = y_test - result['predictions']
                        
                        fig, ax = plt.subplots(figsize=(6, 5))
                        ax.scatter(result['predictions'], residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
                        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
                        ax.set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
                        ax.set_ylabel('Residuals', fontsize=11, fontweight='bold')
                        ax.set_title(f'{model_name} Residuals', fontsize=12, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        plt.close()
                
                # Download predictions
                st.markdown("---")
                st.markdown("## 💾 Download Predictions")
                
                # Create download dataframe
                download_df = pd.DataFrame({
                    'Actual_Yield': y_test.values
                })
                
                for model_name, result in results.items():
                    download_df[f'{model_name}_Predicted'] = result['predictions']
                    download_df[f'{model_name}_Error'] = y_test.values - result['predictions']
                
                csv = download_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name=f"{selected_crop}_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Best model summary
                st.markdown("---")
                st.markdown("## 🏆 Best Model Summary")
                
                best_model = max(results.items(), key=lambda x: x[1]['r2'])
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"### **{best_model[0]}**")
                    st.metric("Best R² Score", f"{best_model[1]['r2']:.4f}")
                    st.metric("MAE", f"{best_model[1]['mae']:.2f} hg/ha")
                    st.metric("RMSE", f"{best_model[1]['rmse']:.2f} hg/ha")
                
                with col2:
                    st.markdown("### Model Insights")
                    st.info(f"""
                    The **{best_model[0]}** model achieved the highest R² score of **{best_model[1]['r2']:.4f}**, 
                    indicating it explains approximately **{best_model[1]['r2']*100:.1f}%** of the variance in crop yield.
                    
                    - **Mean Absolute Error**: {best_model[1]['mae']:.2f} hg/ha
                    - **Root Mean Squared Error**: {best_model[1]['rmse']:.2f} hg/ha
                    - **Training Time**: {best_model[1]['train_time']:.3f} seconds
                    
                    This model provides the most accurate predictions for **{selected_crop}** yield based on 
                    environmental factors including rainfall, temperature, and pesticide usage.
                    """)
    
    with tab2:
        st.markdown("## 📊 Dataset Analysis")
        
        # Get analysis
        analysis = analyze_dataset(df)
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Samples", analysis['total_samples'])
        with col2:
            st.metric("🌱 Unique Crops", len(analysis['crops']))
        with col3:
            st.metric("🌍 Countries", len(analysis['countries']))
        with col4:
            st.metric("📅 Years Covered", f"{analysis['year_range'][0]}-{analysis['year_range'][1]}")
        
        st.markdown("---")
        
        # Yield statistics
        st.markdown("### 🌾 Crop Yield Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Yield", f"{analysis['yield_stats']['mean']:.0f} hg/ha")
        with col2:
            st.metric("Std Dev", f"{analysis['yield_stats']['std']:.0f} hg/ha")
        with col3:
            st.metric("Min Yield", f"{analysis['yield_stats']['min']:.0f} hg/ha")
        with col4:
            st.metric("Max Yield", f"{analysis['yield_stats']['max']:.0f} hg/ha")
        
        st.markdown("---")
        
        # Feature distributions
        st.markdown("### 📈 Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['hg/ha_yield'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_xlabel('Yield (hg/ha)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title('Distribution of Crop Yield', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['average_rain_fall_mm_per_year'], bins=50, alpha=0.7, color='green', edgecolor='black')
            ax.set_xlabel('Rainfall (mm/year)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title('Distribution of Annual Rainfall', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
            plt.close()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['pesticides_tonnes'], bins=50, alpha=0.7, color='orange', edgecolor='black')
            ax.set_xlabel('Pesticides (tonnes)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title('Distribution of Pesticide Usage', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['avg_temp'], bins=50, alpha=0.7, color='red', edgecolor='black')
            ax.set_xlabel('Temperature (°C)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title('Distribution of Average Temperature', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # Correlation analysis
        st.markdown("### 🔗 Feature Correlations")
        
        features = ['hg/ha_yield', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
        corr_matrix = df[features].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                   square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # Top crops by yield
        st.markdown("### 🏆 Top Crops by Average Yield")
        
        top_crops = df.groupby('Item')['hg/ha_yield'].mean().sort_values(ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_crops)))
        bars = ax.barh(range(len(top_crops)), top_crops.values, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_yticks(range(len(top_crops)))
        ax.set_yticklabels(top_crops.index)
        ax.set_xlabel('Average Yield (hg/ha)', fontsize=11, fontweight='bold')
        ax.set_title('Top 10 Crops by Average Yield', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.0f}', ha='left', va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with tab3:
        st.markdown("## ℹ️ About This Application")
        
        st.markdown("""
        ### 🎯 Purpose
        This application compares **Classical Machine Learning** models with **Quantum-Inspired** models 
        for predicting crop yield based on environmental factors.
        
        ### 🤖 Models Compared
        
        1. **Random Forest** - An ensemble learning method that combines multiple decision trees
           - Fast training and prediction
           - Provides feature importance scores
           - Robust to outliers
        
        2. **Support Vector Regression (SVR)** - A classical ML algorithm based on support vector machines
           - Effective for non-linear regression
           - Uses RBF kernel for complex relationships
           - Good generalization
        
        3. **Quantum-Inspired SVR** - A quantum-inspired kernel approach
           - Uses periodic feature encoding inspired by quantum circuits
           - Avoids expensive quantum circuit evaluations
           - Maintains quantum principles while being computationally efficient
        
        ### 📊 Features Used
        - **Year** - Temporal information
        - **Rainfall** - Annual precipitation in mm
        - **Pesticides** - Pesticide usage in tonnes
        - **Temperature** - Average annual temperature in °C
        
        ### 📈 Metrics Explained
        - **R² Score** - Coefficient of determination (0-1, higher is better)
        - **MAE** - Mean Absolute Error (average prediction error)
        - **RMSE** - Root Mean Squared Error (penalizes larger errors)
        - **Training Time** - Time taken to train the model
        
        ### 🔬 Dataset
        - **Total Samples**: {0:,}
        - **Crops**: {1}
        - **Countries**: {2}
        - **Time Period**: {3}-{4}
        
        ### 💡 Key Insights
        - Different crops respond differently to environmental factors
        - Quantum-inspired approaches can provide competitive performance
        - Classical ML models remain highly effective for agricultural prediction
        """.format(
            analysis['total_samples'],
            len(analysis['crops']),
            len(analysis['countries']),
            analysis['year_range'][0],
            analysis['year_range'][1]
        ))

else:
    st.error("❌ Unable to load data. Please check that yield_df.csv exists in the project directory.")
    st.info("""
    **Required file:** `yield_df.csv`
    
    **Expected columns:**
    - Area (country/region)
    - Item (crop type)
    - Year
    - hg/ha_yield (target variable)
    - average_rain_fall_mm_per_year
    - pesticides_tonnes
    - avg_temp
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Crop Yield Prediction System</strong> | Classical ML vs Quantum-Inspired ML Comparison</p>
    <p>Built with Streamlit, Scikit-learn, and Quantum-Inspired Algorithms</p>
</div>
""", unsafe_allow_html=True)
