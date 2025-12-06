#!/bin/bash
if [ -n "$VIRTUAL_ENV" ]; then
  deactivate
fi
source venv/bin/activate
python src/robot.py
