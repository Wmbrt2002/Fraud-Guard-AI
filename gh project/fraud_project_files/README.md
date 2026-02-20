Fraud Detection Project - Minimal packaged files
----------------------------------------------
This archive contains runnable .py scripts and an environment.yml to create a conda environment.

Steps to use:
1. Create conda environment:
    conda env create -f environment.yml
2. Activate environment:
    conda activate fraud_env
3. Put your dataset CSV in data/raw/creditcard.csv (or change path in TRAIN script)
4. Run training:
    python src/train.py
5. Start API:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
6. Start dashboard:
    streamlit run src/dashboard.py
