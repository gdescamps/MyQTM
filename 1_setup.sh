#!/bin/bash
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
pip install -U packaging==23.2 setuptools==75.8.0 wheel ninja
pip install -r requirements.txt

pip install -e .

echo "Please wait until DVC finishes, it will take a while..."
dvc pull -r gcs outputs.dvc 
dvc pull -r gcs llm_cache.pkl.dvc
dvc pull -r gcs data/fmp_data.dvc
