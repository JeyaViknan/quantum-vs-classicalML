# Installation Guide

## Prerequisites
- Python 3.8 or higher
- pip package manager

## Step-by-Step Installation

### 1. Download and Extract
Download the ZIP file and extract it to your desired location.

### 2. Navigate to Project Directory
\`\`\`bash
cd path/to/extracted/folder
\`\`\`

### 3. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

**Note:** Qiskit installation may take a few minutes as it has several dependencies.

### 4. Verify Installation
You can verify Qiskit is installed correctly by running:
\`\`\`bash
python -c "import qiskit; print(qiskit.__version__)"
\`\`\`

### 5. Run the Application
\`\`\`bash
streamlit run crop_yield_ml_comparison.py
\`\`\`

The app will open in your browser at `http://localhost:8501`

## Troubleshooting

### Qiskit Installation Issues

If you encounter issues installing Qiskit, try:

1. **Update pip:**
   \`\`\`bash
   pip install --upgrade pip
   \`\`\`

2. **Install packages individually:**
   \`\`\`bash
   pip install qiskit
   pip install qiskit-machine-learning
   pip install qiskit-algorithms
   \`\`\`

3. **Use a virtual environment (recommended):**
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   \`\`\`

### Common Errors

**"No module named 'qiskit'"**
- Make sure you installed the requirements: `pip install -r requirements.txt`
- Verify you're using the correct Python environment

**"ImportError: cannot import name 'QSVR'"**
- Update qiskit-machine-learning: `pip install --upgrade qiskit-machine-learning`

**Quantum model is slow**
- This is expected! Quantum simulation is computationally intensive
- The app uses a subset of 500 samples for quantum training to speed things up

## System Requirements

- **RAM:** Minimum 4GB (8GB recommended for quantum models)
- **CPU:** Multi-core processor recommended
- **OS:** Windows, macOS, or Linux
</parameter>
