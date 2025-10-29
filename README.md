# 🌾 Crop Yield Prediction: Classical vs Quantum ML

A comprehensive Streamlit application comparing classical machine learning models (Random Forest, SVR) with quantum machine learning (Quantum SVR) for agricultural crop yield prediction.

## ✨ Features

### 🤖 Multiple ML Models
- **Random Forest Regressor** - Ensemble learning with customizable parameters
- **Support Vector Regression (SVR)** - Kernel-based regression with RBF kernel
- **Quantum SVR** - Quantum-enhanced support vector regression using Qiskit

### 📊 Comprehensive Analysis
- **Performance Metrics**: R² Score, MAE, RMSE, Training Time
- **Cross-Validation**: 5-fold CV scores for robust evaluation
- **Feature Importance**: Understand which factors drive predictions
- **Residual Analysis**: Visualize prediction errors and patterns

### 📈 Rich Visualizations
- Actual vs Predicted scatter plots
- Error distribution histograms
- Feature importance bar charts
- Residual plots for error analysis

### 🎛️ Interactive Controls
- Crop selection from multiple varieties
- Model selection (enable/disable individual models)
- Advanced hyperparameter tuning
- Test set size and random state configuration

### 💾 Export Capabilities
- Download predictions as CSV
- Export includes actual values, predictions, and errors for all models

## 🚀 Installation

### 1. Download the Project
Click the three dots (⋯) in the top right and select "Download ZIP"

### 2. Install Dependencies

**Basic Installation (without Quantum ML):**
\`\`\`bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
\`\`\`

**Full Installation (with Quantum ML):**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. Run the Application
\`\`\`bash
streamlit run crop_yield_ml_comparison.py
\`\`\`

The app will open in your browser at `http://localhost:8501`

## 📁 Required Data Files

The application requires `yield_df.csv` with the following columns:
- `Area` - Country/region name
- `Item` - Crop type (e.g., Maize, Wheat, Rice)
- `Year` - Year of observation
- `hg/ha_yield` - Crop yield in hectograms per hectare (target variable)
- `average_rain_fall_mm_per_year` - Annual rainfall in mm
- `pesticides_tonnes` - Pesticide usage in tonnes
- `avg_temp` - Average temperature in °C

## 🎯 Usage Guide

### Basic Workflow
1. **Select Crop**: Choose from available crops in the sidebar
2. **Choose Models**: Enable the models you want to compare
3. **Adjust Settings**: (Optional) Fine-tune hyperparameters in Advanced Settings
4. **Run Analysis**: Click "Run Analysis" to train models and view results
5. **Review Results**: Examine metrics, visualizations, and insights
6. **Download Data**: Export predictions for further analysis

### Model Configuration

**Random Forest Parameters:**
- Number of Trees: 50-500 (default: 100)
- Max Depth: 5-30 (default: 10)

**SVR Parameters:**
- C Parameter: 0.1-10.0 (default: 1.0)
- Epsilon: 0.01-1.0 (default: 0.1)

**General Settings:**
- Test Set Size: 10-40% (default: 20%)
- Random State: 0-100 (default: 42)

## 🔬 Understanding the Results

### Performance Metrics

**R² Score (Coefficient of Determination)**
- Range: -∞ to 1.0 (higher is better)
- Interpretation: Proportion of variance explained by the model
- Good: > 0.7, Excellent: > 0.9

**MAE (Mean Absolute Error)**
- Average absolute difference between predictions and actual values
- Same units as target variable (hg/ha)
- Lower is better

**RMSE (Root Mean Squared Error)**
- Square root of average squared errors
- Penalizes large errors more than MAE
- Lower is better

**Training Time**
- Time taken to train the model in seconds
- Quantum models typically take longer

### Cross-Validation Scores
- 5-fold CV provides robust performance estimates
- Mean ± Standard Deviation shows consistency
- Lower std indicates more stable predictions

## ⚛️ Quantum Machine Learning

The Quantum SVR model uses:
- **ZZFeatureMap**: Quantum feature encoding with 2 repetitions
- **FidelityQuantumKernel**: Quantum kernel for similarity computation
- **Qiskit Sampler**: Quantum circuit execution

**Note**: Quantum models are computationally intensive and may take longer to train, especially with large datasets.

## 🐛 Troubleshooting

### Qiskit Installation Issues

If you encounter Qiskit installation problems:

\`\`\`bash
# Uninstall existing versions
pip uninstall qiskit qiskit-machine-learning qiskit-algorithms -y

# Install specific compatible versions
pip install qiskit==1.0.0 qiskit-machine-learning==0.7.0 qiskit-algorithms==0.3.0
\`\`\`

### Data Loading Errors

Ensure:
- `yield_df.csv` is in the same directory as the Python script
- CSV file has all required columns
- No missing values in critical columns

### Performance Issues

For large datasets:
- Reduce the number of Random Forest trees
- Use a smaller test set size
- Disable cross-validation for quantum models

## 📊 Sample Results

Typical performance on agricultural datasets:
- **Random Forest**: R² ~ 0.85-0.95, Fast training
- **SVR**: R² ~ 0.80-0.90, Moderate training time
- **Quantum SVR**: R² ~ 0.75-0.90, Slower training

## 🤝 Contributing

This is an educational and research tool. Feel free to:
- Add new ML models
- Enhance visualizations
- Improve quantum algorithms
- Add more evaluation metrics

## 📝 License

Open source - feel free to use and modify for your projects.

## 🙏 Acknowledgments

Built with:
- **Streamlit** - Web application framework
- **Scikit-learn** - Classical ML algorithms
- **Qiskit** - Quantum computing framework
- **Matplotlib/Seaborn** - Data visualization

---

**Happy Predicting! 🌾📊**
