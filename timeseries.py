# ======================================================================
# ADVANCED TIME SERIES FORECASTING WITH DEEP LEARNING & ATTENTION (ONE PAGE)
# ======================================================================

import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# ======================================================================
# 1. COMPLEX MULTIVARIATE DATASET GENERATION
# ======================================================================
def generate_multivariate_dataset():
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", "2022-12-31", freq="D")
    n = len(dates)

    trend = 0.01 * np.arange(n)
    season = 10 * np.sin(2 * np.pi * np.arange(n) / 365)
    weekly = 5 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = np.random.normal(0, 2, n)

    main_series = trend + season + weekly + noise
    aux1 = np.roll(main_series, 3) + np.random.normal(0, 1, n)
    aux2 = np.roll(main_series, 7) + np.random.normal(0, 1, n)

    df = pd.DataFrame({
        "ds": dates,
        "y": main_series,
        "x1": aux1,
        "x2": aux2
    })
    return df

# ======================================================================
# 2. SEQ2SEQ WITH BAHNADAU ATTENTION
# ======================================================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        outputs, (h, c) = self.lstm(x)
        return outputs, (h, c)


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        score = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden.unsqueeze(1))))
        attention_weights = torch.softmax(score, dim=1)
        context = torch.sum(attention_weights * encoder_outputs, dim=1)
        return context, attention_weights


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.attn = BahdanauAttention(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + output_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, y_prev, hidden, cell, encoder_outputs):
        context, weights = self.attn(hidden.squeeze(0), encoder_outputs)
        lstm_input = torch.cat([y_prev, context], dim=1).unsqueeze(1)
        out, (h, c) = self.lstm(lstm_input, (hidden, cell))
        pred = self.fc(out.squeeze(1))
        return pred, h, c, weights


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, horizon):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, output_dim)
        self.horizon = horizon

    def forward(self, x):
        encoder_outputs, (h, c) = self.encoder(x)
        y_prev = torch.zeros(x.size(0), 1)  
        predictions = []
        attentions = []

        for _ in range(self.horizon):
            pred, h, c, w = self.decoder(y_prev, h, c, encoder_outputs)
            predictions.append(pred)
            attentions.append(w.squeeze(-1))
            y_prev = pred  

        return torch.stack(predictions, dim=1), torch.stack(attentions, dim=1)

# ======================================================================
# 3. TRAINING, CROSS-VALIDATION & BASELINE
# ======================================================================
def create_sequences(df, seq_len, horizon):
    data = df[["y", "x1", "x2"]].values
    seqs, targs = [], []
    for i in range(len(data) - seq_len - horizon):
        seqs.append(data[i:i+seq_len])
        targs.append(data[i+seq_len:i+seq_len+horizon, 0])
    return np.array(seqs), np.array(targs)

def evaluate_baseline(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred))
    }

def train_model(df):
    seq_len, horizon = 30, 7
    seqs, targs = create_sequences(df, seq_len, horizon)

    scaler = StandardScaler()
    seqs = scaler.fit_transform(seqs.reshape(-1, 3)).reshape(seqs.shape)

    X, y = torch.tensor(seqs, dtype=np.float32), torch.tensor(targs, dtype=torch.float32)

    tscv = TimeSeriesSplit(n_splits=3)
    hidden_dim = 64
    all_metrics = []

    for train_idx, test_idx in tscv.split(X):
        model = Seq2Seq(input_dim=3, hidden_dim=hidden_dim, output_dim=1, horizon=horizon)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for epoch in range(15):
            opt.zero_grad()
            preds, _ = model(X_train)
            loss = loss_fn(preds.squeeze(-1), y_train)
            loss.backward()
            opt.step()

        preds, attn = model(X_test)
        preds = preds.squeeze(-1).detach().numpy()

        metrics = evaluate_baseline(y_test.numpy().flatten(), preds.flatten())
        all_metrics.append(metrics)

    return model, all_metrics, attn[-1]


# ======================================================================
# 4. DRIVER: RUN EVERYTHING
# ======================================================================
if __name__ == "__main__":
    print("\n=== Generating Dataset ===")
    df = generate_multivariate_dataset()
    df.to_csv("multivariate_time_series.csv", index=False)
    print(df.head())

    print("\n=== Training Seq2Seq with Attention ===")
    model, metrics, attention = train_model(df)

    print("\n=== Cross-Validation Metrics ===")
    for i, m in enumerate(metrics):
        print(f"Fold {i+1}: MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}")

    print("\n=== Attention Weights (Last Forecast Step) ===")
    print(attention.detach().numpy())

    print("\n=== Project Completed Successfully ===")
