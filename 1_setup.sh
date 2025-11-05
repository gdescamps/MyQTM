#!/bin/bash
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
pip install -U packaging setuptools wheel ninja
pip install -r requirements.txt
pip install -e .
dvc pull
