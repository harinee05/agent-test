#!/bin/bash

# Activate Azure's auto-created virtual environment (covers most platforms)
if [ -d "/antenv" ]; then
    source /antenv/bin/activate
elif [ -d "/home/site/wwwroot/antenv" ]; then
    source /home/site/wwwroot/antenv/bin/activate
elif [ -d "/home/pythonenv" ]; then
    source /home/pythonenv/3.11/bin/activate
elif [ -d "/tmp/oryx-venv" ]; then
    source /tmp/oryx-venv/bin/activate
fi

uvicorn model_api:app --host 0.0.0.0 --port 9000 &
streamlit run svc_train.py --server.port 8000 --server.address 0.0.0.0
