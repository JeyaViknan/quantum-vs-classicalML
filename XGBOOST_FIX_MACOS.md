# XGBoost macOS Installation Fix

## Problem
You're getting: `Library not loaded: @rpath/libomp.dylib`

This happens because XGBoost requires OpenMP runtime on macOS.

## Solution

### Option 1: Install OpenMP (Recommended)
\`\`\`bash
brew install libomp
\`\`\`

Then reinstall XGBoost:
\`\`\`bash
pip uninstall xgboost -y
pip install xgboost
\`\`\`

### Option 2: Use Conda (Alternative)
\`\`\`bash
conda install -c conda-forge xgboost
\`\`\`

### Option 3: Skip XGBoost
The app will work fine without XGBoost - just uncheck it in the Model Selection.

## Verification
After installation, test with:
\`\`\`python
import xgboost as xgb
print(xgboost.__version__)
\`\`\`

## If Still Having Issues
1. Make sure Homebrew is up to date: `brew update`
2. Verify libomp installation: `brew list libomp`
3. Try: `pip install --upgrade --force-reinstall xgboost`
