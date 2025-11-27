#!/bin/bash

source venv/bin/activate

# Check status and add outputs only if necessary
if dvc status outputs 2>&1 | grep -q "modified"; then
  dvc add outputs
  git commit -m 'update outputs'
  git push
  dvc push
else
  echo "DVC outputs is already up to date"
fi

# Check status and add llm_cache.pkl only if necessary
if dvc status llm_cache.pkl 2>&1 | grep -q "modified"; then
  dvc add llm_cache.pkl
  git commit -m 'update llm_cache.pkl'
  git push
  dvc push
else
  echo "DVC llm_cache.pkl is already up to date"
fi

pushd data
# Check status and add fmp_data only if necessary
if dvc status fmp_data 2>&1 | grep -q "modified"; then
  dvc add fmp_data
  git commit -m 'update fmp_data'
  git push
  dvc push
else
  echo "DVC fmp_data is already up to date"
fi
popd