# =====================================================================
# ADVANCED TIME SERIES FORECASTING WITH PROPHET & N-BEATS (SINGLE FILE)
# =====================================================================

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. SYNTHETIC DATASET GENERATION
# ============================================================
def generate_synthetic_timeseries():
    np.random.seed(42)
    dates = pd.date_range("2019-01-01", "2022-12-31", freq="D")
    n = len(dates)

    trend = 0.02 * np.arange(n)
    trend[n // 2:] += 30  # change-point

    weekly = 10 * np.sin(2 * np.pi * np.arange(n) / 7)
    yearly = 20 * np.sin(2 * np.pi * np.arange(n) / 365)
    noise = np.random.normal(scale=5, size=n)

    spikes = np.zeros(n)
    spike_idx = np.random.choice(n, 12, replace=False)
    spikes[spike_idx] += np.random.uniform(15, 40, len(spike_idx))

    y = trend + weekly + yearly + noise + spikes
    df = pd.DataFrame({"ds": dates, "y": y})
    return df


# ============================================================
# 2. PROPHET MODEL BASELINE
# ============================================================
def train_prophet(df):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.5
    )

    model.add_country_holidays("US")
    model.fit(df)
    return model


def evaluate_prophet(model):
    df_cv = cross_validation(
        model,
        initial="730 days",
        horizon="90 days",
        period="180 days"
    )
    df_perf = performance_metrics(df_cv)
    return df_perf


# ============================================================
# 3. N-BEATS MODEL
# ============================================================
class NBeatsBlock(nn.Module):
    def __init__(self, input_size, hidden_size, theta_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.theta = nn.Linear(hidden_size, theta_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.theta(x)


class NBeats(nn.Module):
    def __init__(self, backcast_len=30, forecast_len=7):
        super().__init__()
        self.backcast_len = backcast_len
        self.forecast_len = forecast_len
        self.block = NBeatsBlock(
            input_size=backcast_len,
            hidden_size=128,
            theta_size=backcast_len + forecast_len,
        )

    def forward(self, x):
        theta = self.block(x)
        backcast = theta[:, :self.backcast_len]
        forecast = theta[:, self.backcast_len:]
        return backcast, forecast


class SeriesDataset(Dataset):
    def __init__(self, series, input_len, output_len):
        self.series = series
        self.in_len = input_len
        self.out_len = output_len

    def __len__(self):
        return len(self.series) - self.in_len - self.out_len

    def __getitem__(self, idx):
        x = self.series[idx:idx + self.in_len]
        y = self.series[idx + self.in_len:idx + self.in_len + self.out_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train_nbeats(df):
    series = df["y"].values.reshape(-1, 1)
    scaler = StandardScaler()
    series = scaler.fit_transform(series).flatten()

    model = NBeats(30, 7)
    opt = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.MSELoss()

    dataset = SeriesDataset(series, 30, 7)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(10):
        for x, y in loader:
            _, f = model(x)
            loss = loss_fn(f, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    torch.save(model.state_dict(), "nbeats_weights.pth")
    return model


# ============================================================
# 4. DRIVER CODE (RUN EVERYTHING)
# ============================================================
if __name__ == "__main__":

    print("\n=== Generating dataset ===")
    df = generate_synthetic_timeseries()
    df.to_csv("synthetic_time_series.csv", index=False)
    print(df.head())

    print("\n=== Training Prophet ===")
    prophet_model = train_prophet(df)
    prophet_perf = evaluate_prophet(prophet_model)
    print("\nProphet Metrics:")
    print(prophet_perf[["mae", "rmse", "mape"]].head())

    print("\n=== Training N-BEATS ===")
    nbeats_model = train_nbeats(df)
    print("N-BEATS training complete. Weights saved to nbeats_weights.pth")

    print("\n=== PROJECT COMPLETE ===")
