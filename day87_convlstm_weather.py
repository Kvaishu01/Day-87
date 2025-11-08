# day87_convlstm_weather.py
"""
Day 87 — ConvLSTM Weather Prediction (Synthetic dataset)
- Generates synthetic spatio-temporal "weather" (moving Gaussian blobs).
- Trains a small ConvLSTM2D model to predict the next frame from a short history.
- Streamlit UI to run training and visualize predictions.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf

st.set_page_config(page_title="ConvLSTM Weather Prediction", layout="centered")
st.title("🌦️ Day 87 — ConvLSTM Weather Prediction")

st.markdown(
    """
This demo trains a small ConvLSTM model on a synthetic dataset (moving Gaussian blobs)
to predict the next spatial frame from previous frames. Use the controls below to run.
"""
)

# -------------------------
# Utilities — synthetic data
# -------------------------
def make_gaussian_grid(cx, cy, sigma, rows, cols):
    """Return an (rows,cols) Gaussian centered at (cx,cy)."""
    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)
    xv, yv = np.meshgrid(x, y)
    g = np.exp(-((xv - cx)**2 + (yv - cy)**2) / (2 * sigma**2))
    return g

def generate_moving_blobs(n_samples=300, timesteps=5, rows=16, cols=16, n_blobs=1, random_state=42):
    """
    Create synthetic spatio-temporal samples:
    For each sample, place n_blobs Gaussian blobs that move slightly across timesteps.
    Returns X of shape (n_samples, timesteps, rows, cols, 1) and y of shape (n_samples, rows, cols, 1)
    where y is the *next* frame after the last input timestep.
    """
    rng = np.random.RandomState(random_state)
    X = np.zeros((n_samples, timesteps, rows, cols, 1), dtype=np.float32)
    y = np.zeros((n_samples, rows, cols, 1), dtype=np.float32)

    for i in range(n_samples):
        # initial centers and velocities
        centers = rng.rand(n_blobs, 2)  # each row: (cx, cy) in [0,1]
        vel = (rng.rand(n_blobs, 2) - 0.5) * 0.08  # small motion per step
        sigmas = 0.05 + rng.rand(n_blobs) * 0.08

        frames = []
        for t in range(timesteps + 1):  # +1 because we need next frame as y
            frame = np.zeros((rows, cols), dtype=np.float32)
            for b in range(n_blobs):
                cx, cy = centers[b]
                sigma = sigmas[b]
                g = make_gaussian_grid(cx, cy, sigma, rows, cols)
                frame += g
            # add slight noise
            frame = frame + rng.normal(scale=0.01, size=frame.shape)
            frame = np.clip(frame, 0.0, None)
            frames.append(frame)
            # update centers for next timestep
            centers = centers + vel
            # bounce off boundaries
            for b in range(n_blobs):
                for d in range(2):
                    if centers[b, d] < 0:
                        centers[b, d] = -centers[b, d]
                        vel[b, d] = -vel[b, d]
                    if centers[b, d] > 1:
                        centers[b, d] = 2 - centers[b, d]
                        vel[b, d] = -vel[b, d]

        # Normalize per sample
        frames = np.array(frames)
        frames = frames / (frames.max() + 1e-8)
        X[i] = frames[:timesteps, :, :, np.newaxis]
        y[i] = frames[timesteps, :, :, np.newaxis]

    return X, y

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Dataset & Model Settings")
n_samples = st.sidebar.slider("Number of samples", 100, 1000, 300, step=50)
timesteps = st.sidebar.slider("Input timesteps", 3, 8, 5)
rows = cols = st.sidebar.slider("Spatial grid size (rows=cols)", 8, 32, 16, step=4)
n_blobs = st.sidebar.slider("Number of moving blobs", 1, 3, 1)
test_size = st.sidebar.slider("Test set fraction (%)", 5, 30, 15) / 100.0

st.sidebar.header("Training")
batch_size = st.sidebar.selectbox("Batch size", [8, 16, 32], index=1)
epochs = st.sidebar.slider("Epochs (small for demo)", 1, 30, 8)

# -------------------------
# Generate dataset (lazy)
# -------------------------
if st.button("🔁 Generate Dataset (Synthetic)"):
    with st.spinner("Generating dataset..."):
        X, y = generate_moving_blobs(n_samples=n_samples, timesteps=timesteps,
                                     rows=rows, cols=cols, n_blobs=n_blobs, random_state=42)
        st.session_state["X"] = X
        st.session_state["y"] = y
    st.success("Dataset generated and stored in session.")

if "X" not in st.session_state:
    st.info("Click **Generate Dataset** to create the synthetic weather data.")
    st.stop()

X = st.session_state["X"]
y = st.session_state["y"]
st.write(f"Dataset shapes — X: {X.shape}, y: {y.shape}")

# Show a few example input frames
sample_idx = 0
st.subheader("Example input sequence (first sample)")
fig, axes = plt.subplots(1, timesteps, figsize=(3*timesteps, 3))
for t in range(timesteps):
    ax = axes[t]
    ax.imshow(X[sample_idx, t, :, :, 0], cmap="viridis")
    ax.set_title(f"t={t}")
    ax.axis("off")
st.pyplot(fig)

# -------------------------
# Build ConvLSTM model
# -------------------------
def build_model(timesteps, rows, cols, channels=1):
    model = Sequential()
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                         activation='relu',
                         padding='same',
                         return_sequences=False,
                         input_shape=(timesteps, rows, cols, channels)))
    model.add(BatchNormalization())
    # output a single frame
    model.add(Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same'))
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
    return model

st.subheader("Model")
st.write("Small ConvLSTM2D -> BatchNorm -> Conv2D model (predicts one next frame).")
if st.button("🧠 Build & Train Model"):
    with st.spinner("Preparing data & training... (this may take a few moments)"):
        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model = build_model(timesteps, rows, cols, channels=1)
        history = model.fit(X_train, y_train,
                            validation_data=(X_test, y_test),
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1)
        st.session_state["model"] = model
        st.session_state["history"] = history.history
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
    st.success("Training complete and model saved to session.")

if "model" not in st.session_state:
    st.info("Build and train the model to see predictions.")
    st.stop()

model = st.session_state["model"]
history = st.session_state["history"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]

# Show training curve
st.subheader("Training curve (loss)")
fig2, ax2 = plt.subplots()
ax2.plot(history['loss'], label='train loss')
if 'val_loss' in history:
    ax2.plot(history['val_loss'], label='val loss')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MSE Loss")
ax2.legend()
st.pyplot(fig2)

# Evaluate and display sample predictions
mse_test = model.evaluate(X_test, y_test, verbose=0)
st.write(f"🔍 Test MSE: {mse_test:.6f}")

st.subheader("Sample Predictions (compare true next-frame vs predicted)")
n_show = st.slider("Which test sample index to visualize", 0, max(0, X_test.shape[0]-1), 0)
pred = model.predict(X_test[n_show:n_show+1])
true_frame = y_test[n_show, :, :, 0]

fig3, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(np.mean(X_test[n_show], axis=0)[:, :, 0], cmap='viridis')
ax[0].set_title("Mean of Input Sequence")
ax[0].axis("off")
ax[1].imshow(true_frame, cmap='viridis')
ax[1].set_title("True Next Frame")
ax[1].axis("off")
ax[2].imshow(pred[0, :, :, 0], cmap='viridis')
ax[2].set_title("Predicted Next Frame")
ax[2].axis("off")
st.pyplot(fig3)

st.markdown("---")
st.markdown("✅ This is a compact demo showing how ConvLSTM learns spatio-temporal patterns and predicts the next spatial frame.")
st.markdown("You can push this to GitHub as `100DaysML-Day87-ConvLSTM-WeatherPrediction`.")
