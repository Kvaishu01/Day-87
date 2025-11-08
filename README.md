# Day 87 — ConvLSTM Weather Prediction

## Summary
This project demonstrates ConvLSTM (Convolutional LSTM) for spatio-temporal forecasting. 
It uses a synthetic dataset (moving Gaussian blobs) to simulate "weather" patterns and trains a ConvLSTM2D model to predict the next frame.

## Files
- `day87_convlstm_weather.py` — Streamlit app and training script (self-contained).
- (Optional) Add your trained model or adjust parameters in the script.

## How to run
1. Create a virtual environment (recommended).
2. Install requirements:
   pip install -r requirements.txt
3. Run locally:
   streamlit run day87_convlstm_weather.py

## Notes
- Dataset is synthetic for demo purposes. Replace with real spatio-temporal data (satellite grids, radar frames, etc.) for production.
- For real datasets, use larger model capacity and more data, plus longer training.

## Suggested GitHub repo name
`100DaysML-Day87-ConvLSTM-WeatherPrediction`

