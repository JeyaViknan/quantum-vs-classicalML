import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    XGBOOST_ERROR = None
except ImportError as e:
    XGBOOST_AVAILABLE = False
    XGBOOST_ERROR = str(e)
except Exception as e:
    XGBOOST_AVAILABLE = False
    XGBOOST_ERROR = str(e)

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_algorithms.utils import algorithm_globals
    QISKIT_AVAILABLE = True
    QISKIT_ERROR = None
except ImportError as e:
    QISKIT_AVAILABLE = False
    QISKIT_ERROR = str(e)

try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

try:
    from advanced_hybrid_nn import AdvancedHybridTrainer, visualize_quantum_circuit_structure, visualize_quantum_outputs
    ADVANCED_HYBRID_AVAILABLE = True
except ImportError:
    ADVANCED_HYBRID_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Crop Yield ML Comparison - Enhanced",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
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
    
    .stDataFrame {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
    }
    
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
    
    .stButton > button {
        background-color: #238636 !important;
        color: #ffffff !important;
        border: 1px solid #2ea043 !important;
    }
    
    .stButton > button:hover {
        background-color: #2ea043 !important;
    }
    
    .streamlit-expanderHeader {
        background-color: #161b22 !important;
        color: #58a6ff !important;
    }
    
    .stSidebar {
        background-color: #0d1117 !important;
    }
    
    .stSidebar [data-testid="stMarkdownContainer"] {
        color: #e6edf3 !important;
    }
    
    .stSelectbox, .stSlider, .stNumberInput {
        color: #e6edf3 !important;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background-color: #161b22 !important;
    }
    
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
    
    .stMarkdown {
        color: #e6edf3 !important;
    }
    
    .stProgress > div > div > div {
        background-color: #238636 !important;
    }
    
    .stSpinner {
        color: #58a6ff !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #e6edf3 !important;
    }
    
    p, span, div {
        color: #e6edf3 !important;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🌾 Crop Yield Prediction: Advanced ML Comparison</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Compare Classical, Quantum-Inspired, and Advanced ML Models with Hyperparameter Optimization</div>', unsafe_allow_html=True)

class FeatureEngineer:
    """Advanced feature engineering for crop yield prediction"""
    
    @staticmethod
    def create_lag_features(df, group_col, value_col, lags=[1, 2, 3]):
        """Create lag features for time series data"""
        df_sorted = df.sort_values(by=[group_col, 'Year']).copy()
        
        for lag in lags:
            df_sorted[f'{value_col}_lag_{lag}'] = df_sorted.groupby(group_col)[value_col].shift(lag)
        
        return df_sorted.dropna()
    
    @staticmethod
    def create_interaction_features(X, feature_names):
        """Create interaction terms between features"""
        X_interactions = X.copy()
        
        # Rainfall × Temperature interaction
        if 'average_rain_fall_mm_per_year' in feature_names and 'avg_temp' in feature_names:
            rain_idx = feature_names.index('average_rain_fall_mm_per_year')
            temp_idx = feature_names.index('avg_temp')
            X_interactions = np.column_stack([X_interactions, X[:, rain_idx] * X[:, temp_idx]])
        
        # Pesticides × Rainfall interaction
        if 'pesticides_tonnes' in feature_names and 'average_rain_fall_mm_per_year' in feature_names:
            pest_idx = feature_names.index('pesticides_tonnes')
            rain_idx = feature_names.index('average_rain_fall_mm_per_year')
            X_interactions = np.column_stack([X_interactions, X[:, pest_idx] * X[:, rain_idx]])
        
        return X_interactions
    
    @staticmethod
    def create_polynomial_features(X, degree=2):
        """Create polynomial features"""
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        return poly.fit_transform(X)

class HyperparameterOptimizer:
    """Hyperparameter optimization using GridSearchCV and Optuna"""
    
    @staticmethod
    def optimize_random_forest(X_train, y_train, use_optuna=False):
        """Optimize Random Forest hyperparameters"""
        if use_optuna and OPTUNA_AVAILABLE:
            def objective(trial):
                n_estimators = trial.suggest_int('n_estimators', 50, 300)
                max_depth = trial.suggest_int('max_depth', 5, 30)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
                
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    n_jobs=-1
                )
                
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
                return scores.mean()
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=20, show_progress_bar=False)
            return study.best_params
        else:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=-1),
                param_grid,
                cv=3,
                scoring='r2',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_params_
    
    @staticmethod
    def optimize_svr(X_train, y_train, use_optuna=False):
        """Optimize SVR hyperparameters"""
        if use_optuna and OPTUNA_AVAILABLE:
            def objective(trial):
                C = trial.suggest_float('C', 0.1, 100, log=True)
                epsilon = trial.suggest_float('epsilon', 0.01, 1.0)
                gamma = trial.suggest_float('gamma', 0.001, 1.0, log=True)
                
                model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel='rbf')
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
                return scores.mean()
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=20, show_progress_bar=False)
            return study.best_params
        else:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.5],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
            
            grid_search = GridSearchCV(
                SVR(kernel='rbf'),
                param_grid,
                cv=3,
                scoring='r2',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_params_
    
    @staticmethod
    def optimize_xgboost(X_train, y_train, use_optuna=False):
        """Optimize XGBoost hyperparameters"""
        if use_optuna and OPTUNA_AVAILABLE:
            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                }
                
                model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
                return scores.mean()
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=20, show_progress_bar=False)
            return study.best_params
        else:
            param_grid = {
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [50, 100, 200],
                'subsample': [0.7, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.9, 1.0]
            }
            
            grid_search = GridSearchCV(
                xgb.XGBRegressor(random_state=42, verbosity=0),
                param_grid,
                cv=3,
                scoring='r2',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_params_

class QuantumInspiredKernel:
    """Quantum-inspired kernel using periodic feature encoding"""
    def __init__(self, n_qubits=4):
        self.n_qubits = n_qubits
    
    def _encode_features(self, X):
        """Encode features using periodic quantum-inspired encoding"""
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
        
        gamma = 1.0 / X1_encoded.shape[1]
        kernel_matrix = np.exp(-gamma * np.sum((X1_encoded[:, np.newaxis, :] - X2_encoded[np.newaxis, :, :]) ** 2, axis=2))
        return kernel_matrix

class StatisticalAnalyzer:
    """Statistical analysis and bias detection"""
    
    @staticmethod
    def analyze_residuals(y_true, y_pred):
        """Comprehensive residual analysis"""
        residuals = y_true - y_pred
        
        analysis = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'median': np.median(residuals),
            'skewness': (np.mean(residuals) - np.median(residuals)) / (np.std(residuals) + 1e-10),
            'kurtosis': np.mean((residuals - np.mean(residuals))**4) / (np.std(residuals)**4 + 1e-10) - 3
        }
        
        return analysis, residuals
    
    @staticmethod
    def detect_bias(y_true, y_pred, features_df):
        """Detect bias in predictions across feature ranges"""
        residuals = y_true - y_pred
        
        bias_analysis = {}
        for col in features_df.columns:
            quartiles = pd.qcut(features_df[col], q=4, duplicates='drop')
            bias_by_quartile = residuals.groupby(quartiles).mean()
            bias_analysis[col] = bias_by_quartile
        
        return bias_analysis
    
    @staticmethod
    def correlation_analysis(residuals, features_df):
        """Analyze correlation between residuals and features"""
        correlations = {}
        for col in features_df.columns:
            corr = np.corrcoef(residuals, features_df[col])[0, 1]
            correlations[col] = corr
        
        return correlations

# Sidebar configuration
st.sidebar.title("Configuration")
st.sidebar.markdown("---")

# Data loading with caching
@st.cache_data
def load_data():
    """Load and validate the crop yield dataset"""
    try:
        df = pd.read_csv('yield_df.csv', index_col=0)
        
        required_cols = ['Area', 'Item', 'Year', 'hg/ha_yield', 
                        'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
        
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing required columns. Expected: {required_cols}")
            return None
        
        df = df.dropna()
        return df
    except FileNotFoundError:
        st.error("Error: yield_df.csv not found.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data
df = load_data()

if df is not None:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Model Comparison", "Hyperparameter Tuning", "Feature Engineering", "SHAP Analysis", "Statistical Insights"])
    
    with tab1:
        # Crop selection
        crops = sorted(df['Item'].unique())
        selected_crop = st.sidebar.selectbox("Select Crop", crops, index=0)
        
        # Model selection
        st.sidebar.markdown("### Model Selection")
        enable_rf = st.sidebar.checkbox("Random Forest", value=True)
        enable_svr = st.sidebar.checkbox("Support Vector Regression", value=True)
        enable_quantum = st.sidebar.checkbox("Quantum-Inspired SVR", value=True)
        enable_xgboost = st.sidebar.checkbox("XGBoost", value=XGBOOST_AVAILABLE)
        enable_catboost = st.sidebar.checkbox("CatBoost", value=CATBOOST_AVAILABLE)
        enable_mlp = st.sidebar.checkbox("Neural Network (MLP)", value=True)
        enable_hqnn = st.sidebar.checkbox("Hybrid Quantum-Classical NN", value=(TORCH_AVAILABLE and PENNYLANE_AVAILABLE))
        enable_advanced_hybrid = st.sidebar.checkbox("Advanced Hybrid NN (with PCA)", value=(ADVANCED_HYBRID_AVAILABLE and TORCH_AVAILABLE and PENNYLANE_AVAILABLE))
        
        # Advanced settings
        with st.sidebar.expander("Advanced Settings"):
            test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
            random_state = st.number_input("Random State", 0, 100, 42)
            use_feature_engineering = st.checkbox("Enable Feature Engineering", value=True)
            use_hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning", value=False)
            use_optuna = st.checkbox("Use Optuna (if available)", value=False)
        
        st.sidebar.markdown("---")
        run_analysis = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)
        
        # Filter data for selected crop
        crop_df = df[df['Item'] == selected_crop].copy()
        
        # Display dataset info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(crop_df))
        with col2:
            st.metric("Countries", crop_df['Area'].nunique())
        with col3:
            st.metric("Year Range", f"{crop_df['Year'].min()}-{crop_df['Year'].max()}")
        with col4:
            st.metric("Selected Crop", selected_crop)
        
        st.markdown("---")
        
        if run_analysis:
            if not any([enable_rf, enable_svr, enable_quantum, enable_xgboost, enable_catboost, enable_mlp, enable_hqnn, enable_advanced_hybrid]):
                st.warning("Please select at least one model to train.")
            else:
                # Prepare data
                with st.spinner("Preparing data..."):
                    X = crop_df[['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']].copy()
                    y = crop_df['hg/ha_yield'].copy()
                    
                    # Feature engineering
                    if use_feature_engineering:
                        X_original = X.copy()
                        # Add interaction terms
                        X['rainfall_temp_interaction'] = X['average_rain_fall_mm_per_year'] * X['avg_temp']
                        X['pesticides_rainfall_interaction'] = X['pesticides_tonnes'] * X['average_rain_fall_mm_per_year']
                        st.info(f"Feature engineering enabled: {X.shape[1]} features created")
                    
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
                if enable_xgboost and XGBOOST_AVAILABLE:
                    models_to_train.append('XGBoost')
                if enable_catboost and CATBOOST_AVAILABLE:
                    models_to_train.append('CatBoost')
                if enable_mlp:
                    models_to_train.append('Neural Network (MLP)')
                if enable_hqnn and (TORCH_AVAILABLE and PENNYLANE_AVAILABLE):
                    models_to_train.append('Hybrid Quantum-Classical NN')
                if enable_advanced_hybrid and ADVANCED_HYBRID_AVAILABLE and TORCH_AVAILABLE and PENNYLANE_AVAILABLE:
                    models_to_train.append('Advanced Hybrid NN (with PCA)')
                
                total_models = len(models_to_train)
                
                for idx, model_name in enumerate(models_to_train):
                    status_text.text(f"Training {model_name}... ({idx+1}/{total_models})")
                    
                    try:
                        if model_name == 'Random Forest':
                            start_time = time.time()
                            
                            if use_hyperparameter_tuning:
                                best_params = HyperparameterOptimizer.optimize_random_forest(X_train_scaled, y_train, use_optuna)
                                rf_model = RandomForestRegressor(**best_params, random_state=random_state, n_jobs=-1)
                            else:
                                rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1)
                            
                            rf_model.fit(X_train_scaled, y_train)
                            y_pred_rf = rf_model.predict(X_test_scaled)
                            train_time_rf = time.time() - start_time
                            
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
                            start_time = time.time()
                            
                            if use_hyperparameter_tuning:
                                best_params = HyperparameterOptimizer.optimize_svr(X_train_scaled, y_train_scaled, use_optuna)
                                svr_model = SVR(**best_params)
                            else:
                                svr_model = SVR(C=1.0, epsilon=0.1, kernel='rbf', gamma='scale')
                            
                            svr_model.fit(X_train_scaled, y_train_scaled)
                            y_pred_svr_scaled = svr_model.predict(X_test_scaled)
                            y_pred_svr = y_scaler.inverse_transform(y_pred_svr_scaled.reshape(-1, 1)).ravel()
                            train_time_svr = time.time() - start_time
                            
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
                            start_time = time.time()
                            qi_kernel = QuantumInspiredKernel(n_qubits=4)
                            qi_svr_model = SVR(C=1.0, epsilon=0.1, kernel=qi_kernel)
                            qi_svr_model.fit(X_train_scaled, y_train_scaled)
                            y_pred_qi_svr_scaled = qi_svr_model.predict(X_test_scaled)
                            y_pred_qi_svr = y_scaler.inverse_transform(y_pred_qi_svr_scaled.reshape(-1, 1)).ravel()
                            train_time_qi_svr = time.time() - start_time
                            
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
                        
                        elif model_name == 'XGBoost':
                            start_time = time.time()
                            
                            if use_hyperparameter_tuning:
                                best_params = HyperparameterOptimizer.optimize_xgboost(X_train_scaled, y_train, use_optuna)
                                xgb_model = xgb.XGBRegressor(**best_params, random_state=random_state, verbosity=0)
                            else:
                                xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state, verbosity=0)
                            
                            xgb_model.fit(X_train_scaled, y_train)
                            y_pred_xgb = xgb_model.predict(X_test_scaled)
                            train_time_xgb = time.time() - start_time
                            
                            cv_scores_xgb = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5, scoring='r2')
                            
                            results['XGBoost'] = {
                                'model': xgb_model,
                                'predictions': y_pred_xgb,
                                'r2': r2_score(y_test, y_pred_xgb),
                                'mae': mean_absolute_error(y_test, y_pred_xgb),
                                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
                                'train_time': train_time_xgb,
                                'cv_mean': cv_scores_xgb.mean(),
                                'cv_std': cv_scores_xgb.std(),
                                'feature_importance': xgb_model.feature_importances_
                            }
                        
                        elif model_name == 'CatBoost':
                            start_time = time.time()
                            
                            cb_model = cb.CatBoostRegressor(
                                iterations=100,
                                depth=5,
                                learning_rate=0.1,
                                random_state=random_state,
                                verbose=False
                            )
                            
                            cb_model.fit(X_train_scaled, y_train)
                            y_pred_cb = cb_model.predict(X_test_scaled)
                            train_time_cb = time.time() - start_time
                            
                            cv_scores_cb = cross_val_score(cb_model, X_train_scaled, y_train, cv=5, scoring='r2')
                            
                            results['CatBoost'] = {
                                'model': cb_model,
                                'predictions': y_pred_cb,
                                'r2': r2_score(y_test, y_pred_cb),
                                'mae': mean_absolute_error(y_test, y_pred_cb),
                                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_cb)),
                                'train_time': train_time_cb,
                                'cv_mean': cv_scores_cb.mean(),
                                'cv_std': cv_scores_cb.std(),
                                'feature_importance': cb_model.feature_importances_
                            }
                        
                        elif model_name == 'Neural Network (MLP)':
                            start_time = time.time()
                            
                            mlp_model = MLPRegressor(
                                hidden_layer_sizes=(64, 32),
                                max_iter=1000,
                                alpha=0.001,
                                learning_rate='adaptive',
                                learning_rate_init=0.01,
                                early_stopping=True,
                                validation_fraction=0.2,
                                n_iter_no_change=50,
                                random_state=random_state,
                                solver='adam',
                                batch_size=32,
                                warm_start=False
                            )
                            
                            mlp_model.fit(X_train_scaled, y_train_scaled)
                            
                            y_pred_mlp_scaled = mlp_model.predict(X_test_scaled)
                            y_pred_mlp = y_scaler.inverse_transform(y_pred_mlp_scaled.reshape(-1, 1)).ravel()
                            
                            train_time_mlp = time.time() - start_time
                            
                            cv_scores_mlp = cross_val_score(mlp_model, X_train_scaled, y_train_scaled, cv=5, scoring='r2')
                            
                            results['Neural Network (MLP)'] = {
                                'model': mlp_model,
                                'predictions': y_pred_mlp,
                                'r2': r2_score(y_test, y_pred_mlp),
                                'mae': mean_absolute_error(y_test, y_pred_mlp),
                                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_mlp)),
                                'train_time': train_time_mlp,
                                'cv_mean': cv_scores_mlp.mean(),
                                'cv_std': cv_scores_mlp.std()
                            }
                        
                        elif model_name == 'Hybrid Quantum-Classical NN':
                            try:
                                from hybrid_quantum_classical_model import HybridModelTrainer
                                
                                start_time = time.time()
                                
                                trainer = HybridModelTrainer(
                                    n_qubits=4,
                                    n_layers=1,
                                    learning_rate=0.01
                                )
                                
                                hqnn_result = trainer.train(
                                    X_train_scaled, y_train,
                                    X_test_scaled, y_test,
                                    epochs=50,
                                    batch_size=16,
                                    validation_split=0.2
                                )
                                
                                train_time_hqnn = time.time() - start_time
                                
                                results['Hybrid Quantum-Classical NN'] = {
                                    'model': trainer.model,
                                    'predictions': hqnn_result['predictions'],
                                    'r2': hqnn_result['r2'],
                                    'mae': hqnn_result['mae'],
                                    'rmse': hqnn_result['rmse'],
                                    'train_time': train_time_hqnn,
                                    'cv_mean': hqnn_result['r2'],
                                    'cv_std': 0.0,
                                    'history': hqnn_result['history'],
                                    'trainer': trainer,
                                    'X_test_scaled': hqnn_result['X_test_scaled']
                                }
                            
                            except Exception as e:
                                st.error(f"Error training Hybrid Quantum-Classical NN: {str(e)}")
                        
                        elif model_name == 'Advanced Hybrid NN (with PCA)':
                            try:
                                start_time = time.time()
                                
                                trainer = AdvancedHybridTrainer(
                                    n_qubits=4,
                                    n_layers=2,
                                    learning_rate=0.001
                                )
                                
                                cv_result = trainer.train_cv(
                                    X_train, y_train,
                                    n_splits=5,
                                    epochs=50,
                                    batch_size=16
                                )
                                
                                train_time_advanced = time.time() - start_time
                                
                                # Use best fold model for predictions
                                best_fold = max(enumerate(cv_result['fold_results']), 
                                              key=lambda x: x[1]['r2'])
                                best_model = best_fold[1]['model']
                                best_preprocessor = best_fold[1]['preprocessor']
                                
                                # Make predictions on test set
                                X_test_processed, _ = best_preprocessor.transform(X_test_scaled)
                                device = next(best_model.parameters()).device
                                X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32).to(device)
                                best_model.eval()
                                with torch.no_grad():
                                    y_pred_scaled = best_model(X_test_tensor).detach().cpu().numpy().flatten()
                                
                                # Inverse transform predictions back to original scale
                                y_pred_advanced = best_preprocessor.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                                
                                results['Advanced Hybrid NN (with PCA)'] = {
                                    'model': best_model,
                                    'predictions': y_pred_advanced,
                                    'r2': r2_score(y_test, y_pred_advanced),
                                    'mae': mean_absolute_error(y_test, y_pred_advanced),
                                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_advanced)),
                                    'train_time': train_time_advanced,
                                    'cv_mean': cv_result['avg_r2'],
                                    'cv_std': 0.0,
                                    'cv_result': cv_result,
                                    'X_test_processed': X_test_processed,
                                    'preprocessor': best_preprocessor
                                }
                            
                            except Exception as e:
                                st.error(f"Error training Advanced Hybrid NN: {str(e)}")
                    
                    except Exception as e:
                        st.error(f"Error training {model_name}: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / total_models)
                
                status_text.text("Training complete!")
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                st.markdown("## Model Performance Comparison")
                
                comparison_data = []
                for model_name, result in results.items():
                    comparison_data.append({
                        'Model': model_name,
                        'R² Score': f"{result['r2']:.4f}" if result['r2'] is not None else "N/A",
                        'MAE': f"{result['mae']:.2f}" if result['mae'] is not None else "N/A",
                        'RMSE': f"{result['rmse']:.2f}" if result['rmse'] is not None else "N/A",
                        'Training Time (s)': f"{result['train_time']:.3f}",
                        'CV R² (mean±std)': f"{result['cv_mean']:.4f}±{result['cv_std']:.4f}" if result['cv_mean'] is not None else "N/A"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Metrics in columns
                st.markdown("### Detailed Metrics")
                cols = st.columns(min(len(results), 3))
                
                for idx, (model_name, result) in enumerate(results.items()):
                    with cols[idx % 3]:
                        st.markdown(f"**{model_name}**")
                        st.metric("R² Score", f"{result['r2']:.4f}" if result['r2'] is not None else "N/A")
                        st.metric("MAE", f"{result['mae']:.2f}" if result['mae'] is not None else "N/A")
                        st.metric("RMSE", f"{result['rmse']:.2f}" if result['rmse'] is not None else "N/A")
                        st.metric("Training Time", f"{result['train_time']:.3f}s")
                
                st.markdown("---")
                
                # Visualizations
                st.markdown("## Visualizations")
                
                # Actual vs Predicted plots
                st.markdown("### Actual vs Predicted Values")
                
                num_models = len(results)
                cols = st.columns(min(num_models, 3))
                
                for idx, (model_name, result) in enumerate(results.items()):
                    with cols[idx % 3]:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        
                        ax.scatter(y_test, result['predictions'], alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
                        
                        min_val = min(y_test.min(), result['predictions'].min())
                        max_val = max(y_test.max(), result['predictions'].max())
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
                        
                        ax.set_xlabel('Actual Yield (hg/ha)', fontsize=11, fontweight='bold')
                        ax.set_ylabel('Predicted Yield (hg/ha)', fontsize=11, fontweight='bold')
                        ax.set_title(f'{model_name}\nR² = {result["r2"]:.4f}' if result['r2'] is not None else "N/A", fontsize=12, fontweight='bold')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        plt.close()
                
                # Error distribution
                st.markdown("### Error Distribution")
                
                fig, axes = plt.subplots(1, min(num_models, 3), figsize=(6*min(num_models, 3), 5))
                if num_models == 1:
                    axes = [axes]
                elif num_models == 2:
                    axes = list(axes)
                else:
                    axes = list(axes[:min(num_models, 3)])
                
                for idx, (model_name, result) in enumerate(list(results.items())[:3]):
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
                
                # Feature importance for tree-based models
                st.markdown("### Feature Importance")
                
                feature_names = X.columns.tolist()
                
                tree_models = {k: v for k, v in results.items() if 'feature_importance' in v}
                
                if tree_models:
                    cols = st.columns(min(len(tree_models), 2))
                    
                    for idx, (model_name, result) in enumerate(tree_models.items()):
                        with cols[idx % 2]:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            importance = result['feature_importance']
                            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
                            bars = ax.barh(feature_names, importance, color=colors, edgecolor='black', linewidth=1.5)
                            
                            ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
                            ax.set_title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
                            ax.grid(True, alpha=0.3, axis='x')
                            
                            for bar in bars:
                                width = bar.get_width()
                                ax.text(width, bar.get_y() + bar.get_height()/2, 
                                       f'{width:.3f}', ha='left', va='center', fontweight='bold', fontsize=10)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                
                # Loss curves for models with training history
                st.markdown("### Training Loss Curves")
                
                hqnn_models = {k: v for k, v in results.items() if 'history' in v}
                
                if hqnn_models:
                    for model_name, result in hqnn_models.items():
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        history = result['history']
                        epochs_range = range(1, len(history['train_losses']) + 1)
                        
                        ax.plot(epochs_range, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
                        ax.plot(epochs_range, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
                        
                        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
                        ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
                        ax.set_title(f'{model_name} - Training History', fontsize=14, fontweight='bold')
                        ax.legend(fontsize=11)
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        plt.close()
                
                # Download predictions
                st.markdown("---")
                st.markdown("## Download Predictions")
                
                download_df = pd.DataFrame({
                    'Actual_Yield': y_test.values
                })
                
                for model_name, result in results.items():
                    download_df[f'{model_name}_Predicted'] = result['predictions']
                    download_df[f'{model_name}_Error'] = y_test.values - result['predictions']
                
                csv = download_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name=f"{selected_crop}_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Best model summary
                st.markdown("---")
                st.markdown("## Best Model Summary")
                
                best_model = max(results.items(), key=lambda x: x[1]['r2'] if x[1]['r2'] is not None else -np.inf)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"### **{best_model[0]}**")
                    st.metric("Best R² Score", f"{best_model[1]['r2']:.4f}" if best_model[1]['r2'] is not None else "N/A")
                    st.metric("MAE", f"{best_model[1]['mae']:.2f} hg/ha" if best_model[1]['mae'] is not None else "N/A")
                    st.metric("RMSE", f"{best_model[1]['rmse']:.2f} hg/ha" if best_model[1]['rmse'] is not None else "N/A")
                
                with col2:
                    st.markdown("### Model Insights")
                    st.info(f"""
                    The **{best_model[0]}** model achieved the highest R² score of **{best_model[1]['r2']:.4f}**, 
                    explaining approximately **{best_model[1]['r2']*100:.1f}%** of the variance in crop yield.
                    
                    - **Mean Absolute Error**: {best_model[1]['mae']:.2f} hg/ha
                    - **Root Mean Squared Error**: {best_model[1]['rmse']:.2f} hg/ha
                    - **Training Time**: {best_model[1]['train_time']:.3f} seconds
                    
                    This model provides the most accurate predictions for **{selected_crop}** yield.
                    """)
                
                st.markdown("### Quantum Layer Analysis")
                
                hqnn_results = {k: v for k, v in results.items() if 'trainer' in v}
                
                if hqnn_results:
                    for model_name, result in hqnn_results.items():
                        st.markdown(f"#### {model_name}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Quantum circuit structure
                            from hybrid_quantum_classical_model import visualize_quantum_circuit_structure
                            fig = visualize_quantum_circuit_structure(n_qubits=4, n_layers=1)
                            st.pyplot(fig)
                            plt.close()
                        
                        with col2:
                            # Quantum output distribution
                            from hybrid_quantum_classical_model import visualize_quantum_outputs
                            fig = visualize_quantum_outputs(result['model'], result['X_test_scaled'][:50])
                            st.pyplot(fig)
                            plt.close()
                
                advanced_hybrid_results = {k: v for k, v in results.items() if 'trainer' in v and 'Advanced Hybrid NN (with PCA)' in k}
                
                if advanced_hybrid_results:
                    for model_name, result in advanced_hybrid_results.items():
                        st.markdown(f"#### {model_name}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Quantum circuit structure
                            fig = visualize_quantum_circuit_structure(n_qubits=4, n_layers=1)
                            st.pyplot(fig)
                            plt.close()
                        
                        with col2:
                            # Quantum output distribution
                            fig = visualize_quantum_outputs(result['model'], result['X_test_scaled'][:50])
                            st.pyplot(fig)
                            plt.close()
                
                # Advanced Hybrid Model Analysis
                advanced_hybrid_results = {k: v for k, v in results.items() if 'cv_result' in v}
                
                if advanced_hybrid_results:
                    st.markdown("### Advanced Hybrid NN Analysis")
                    
                    for model_name, result in advanced_hybrid_results.items():
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"#### {model_name}")
                            st.metric("Cross-Validation R²", f"{result['cv_mean']:.4f}")
                            st.metric("Test R²", f"{result['r2']:.4f}")
                            st.metric("Test RMSE", f"{result['rmse']:.2f}")
                            
                            # Quantum circuit structure
                            fig = visualize_quantum_circuit_structure(n_qubits=4, n_layers=2)
                            st.pyplot(fig)
                            plt.close()
                        
                        with col2:
                            # Quantum output distribution
                            fig = visualize_quantum_outputs(result['model'], result['X_test_processed'][:50])
                            st.pyplot(fig)
                            plt.close()
                        
                        # Cross-validation fold results
                        st.markdown("#### Fold-wise Performance")
                        fold_data = []
                        for fold_idx, fold_result in enumerate(result['cv_result']['fold_results']):
                            fold_data.append({
                                'Fold': fold_idx + 1,
                                'R² Score': f"{fold_result['r2']:.4f}",
                                'RMSE': f"{fold_result['rmse']:.2f}",
                                'MAE': f"{fold_result['mae']:.2f}"
                            })
                        
                        fold_df = pd.DataFrame(fold_data)
                        st.dataframe(fold_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("## Hyperparameter Tuning")
        st.info("This tab demonstrates hyperparameter optimization capabilities. Enable 'Hyperparameter Tuning' in Advanced Settings to use optimized parameters.")
        
        st.markdown("""
        ### Available Optimization Methods
        
        1. **GridSearchCV** - Exhaustive search over specified parameter values
           - Comprehensive but computationally expensive
           - Good for smaller parameter spaces
        
        2. **Optuna** - Bayesian optimization framework
           - More efficient than grid search
           - Adaptive sampling of parameter space
           - Requires Optuna library
        
        ### Tuned Parameters by Model
        
        **Random Forest:**
        - n_estimators: 50-300
        - max_depth: 5-30
        - min_samples_split: 2-10
        - min_samples_leaf: 1-5
        
        **SVR:**
        - C: 0.1-100 (log scale)
        - epsilon: 0.01-1.0
        - gamma: 0.001-1.0 (log scale)
        
        **XGBoost:**
        - max_depth: 3-10
        - learning_rate: 0.01-0.3
        - n_estimators: 50-300
        - subsample: 0.5-1.0
        - colsample_bytree: 0.5-1.0
        """)
    
    with tab3:
        st.markdown("## Feature Engineering")
        st.info("Feature engineering is automatically applied when enabled in Advanced Settings.")
        
        st.markdown("""
        ### Features Created
        
        **Original Features:**
        - Year
        - Average Rainfall (mm/year)
        - Pesticides (tonnes)
        - Average Temperature (°C)
        
        **Engineered Features:**
        - Rainfall × Temperature Interaction
        - Pesticides × Rainfall Interaction
        
        ### Benefits
        - Captures non-linear relationships
        - Improves model performance
        - Provides domain-specific insights
        
        ### Example Interactions
        - **Rainfall × Temperature**: Captures combined effect of water and heat on crop growth
        - **Pesticides × Rainfall**: Captures pesticide effectiveness under different moisture conditions
        """)
    
    with tab4:
        st.markdown("## SHAP Analysis")
        
        if SHAP_AVAILABLE:
            st.info("SHAP library is available. Run analysis with a tree-based model to see feature importance explanations.")
            st.markdown("""
            SHAP (SHapley Additive exPlanations) provides:
            - Individual prediction explanations
            - Feature importance rankings
            - Feature interaction analysis
            - Model behavior insights
            """)
        else:
            st.warning("SHAP library not installed. Install with: pip install shap")
            st.markdown("""
            SHAP provides advanced model interpretability:
            - Explains individual predictions
            - Shows feature contributions
            - Identifies feature interactions
            - Detects model biases
            """)
    
    with tab5:
        st.markdown("## Statistical Insights")
        st.info("Run analysis to see detailed statistical analysis of model residuals and bias detection.")
        
        st.markdown("""
        ### Residual Analysis
        - Mean and standard deviation of errors
        - Distribution shape (skewness, kurtosis)
        - Outlier detection
        
        ### Bias Detection
        - Bias across feature ranges
        - Systematic over/under-prediction
        - Feature-residual correlations
        
        ### Statistical Tests
        - Normality of residuals
        - Heteroscedasticity detection
        - Autocorrelation analysis
        """)

else:
    st.error("Unable to load data. Please check that yield_df.csv exists.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Enhanced Crop Yield Prediction System</strong> | Advanced ML Comparison with Optimization</p>
</div>
""", unsafe_allow_html=True)
