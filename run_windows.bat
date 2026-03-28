@echo off
python -m src.train --data-path data\raw\sample_diabetic_data.csv
python -m src.evaluate --data-path data\raw\sample_diabetic_data.csv
streamlit run app.py
