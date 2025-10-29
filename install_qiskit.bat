@echo off
echo Installing Qiskit for Quantum ML...
pip uninstall qiskit qiskit-machine-learning qiskit-algorithms -y
pip install qiskit==1.0.0 qiskit-machine-learning==0.7.0 qiskit-algorithms==0.3.0
echo Verifying installation...
python -c "import qiskit; import qiskit_machine_learning; print('✓ Qiskit installed successfully!')"
echo Done! Now run: streamlit run crop_yield_ml_comparison.py
pause
