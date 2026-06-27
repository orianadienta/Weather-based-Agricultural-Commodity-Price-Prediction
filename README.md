# Agricultural Commodity Price Prediction Based on Weather Data

A machine learning study comparing **XGBoost** and **AutoGluon Tabular** for one-day-ahead price forecasting of seven agricultural commodities at Sepinggan Market, Balikpapan. Two configurations are evaluated: a full-weather model and a no-weather baseline, to empirically assess the contribution of weather features to prediction accuracy.

---

## Repository Structure

```
.
├── Price_Prediction_Based_on_the_Weather_ML_C_9ipynb.ipynb   # Full-weather model (XGBoost + AutoGluon)
├── Price_Prediction_Without_Weather_ML_C_9.ipynb              # No-weather baseline (XGBoost + AutoGluon)
└── Weather and price dataset.csv                              # Merged commodity price + weather dataset
```

---

## Dataset

**File:** `Weather and price dataset.csv`  
**Period:** 1 January 2024 to 21 May 2026  
**Observations:** 872 daily records

### Sources

| Data | Source |
|------|--------|
| Commodity prices | [SAHABAT Balikpapan](https://sahabat.balikpapan.go.id) (web scraping) |
| Weather data | [Open-Meteo API](https://open-meteo.com) |

Both sources were merged using the `date` field as the join key.

### Target Commodities (Sepinggan Market)

1. Curly Red Chili (*Cabai Merah Keriting*)
2. Large Red Chili (*Cabai Merah Besar*)
3. Bird's Eye Chili (*Cabai Rawit*)
4. Spinach (*Bayam*)
5. Yardlong Bean (*Kacang Panjang*)
6. Water Spinach (*Kangkung*)
7. Mustard Greens (*Sawi*)

### Dataset Columns

**Weather columns (all recorded):**

| Column | Description | Unit |
|--------|-------------|------|
| `max_temperature_c` | Maximum daily temperature | °C |
| `min_temperature_c` | Minimum daily temperature | °C |
| `mean_temperature_c` | Mean daily temperature | °C |
| `rainfall_mm` | Daily rainfall | mm |
| `precipitation_mm` | Total daily precipitation | mm |
| `rain_duration_hours` | Duration of rainfall | hours |
| `max_wind_speed_kmh` | Maximum wind speed | km/h |
| `wind_speed_kmh` | Mean wind speed | km/h |
| `max_humidity_pct` | Maximum relative humidity | % |
| `min_humidity_pct` | Minimum relative humidity | % |
| `sunshine_duration_seconds` | Sunshine duration (raw) | seconds |
| `sunshine_duration_hours` | Sunshine duration (derived) | hours |
| `wmo_weather_code` | WMO weather classification code | (categorical) |

**Price columns:**

`Curly Chili Pepper`, `Large Red Chili Pepper`, `Bird's Eye Chili Pepper`, `Spinach`, `Long Bean`, `Water Spinach`, `Mustard Greens`

---

## Methodology

### Prediction Target

Each model predicts the price of the following day (H+1) for each commodity. The target variable is constructed by shifting the price column by -1:

```python
df[f'{col}_target_h1'] = df[col].shift(-1)
```

### Feature Engineering

For each lag window configuration N (N = 1 to 7), lag features are generated from H-1 up to H-N:

**Price lag features (both configurations):**

```
{commodity}_lag1, {commodity}_lag2, ..., {commodity}_lagN
```

**Weather lag features (full-weather configuration only):**

Five weather variables are selected as model inputs:

| Variable | Column |
|----------|--------|
| Rainfall | `rainfall_mm` |
| Mean temperature | `mean_temperature_c` |
| Minimum humidity | `min_humidity_pct` |
| Mean wind speed | `wind_speed_kmh` |
| Sunshine duration | `sunshine_duration_hours` |

Lagged versions are generated from H-1 up to H-N:

```
{weather_col}_lag1, {weather_col}_lag2, ..., {weather_col}_lagN
```

`dropna()` is applied after feature generation, so the effective dataset size and train/test boundary shift slightly across different values of N.

### Train/Test Split

Chronological 80/20 split with no shuffling to preserve temporal ordering and prevent data leakage. The split is recalculated per N after `dropna()`.

### Models

| Model | Configuration |
|-------|--------------|
| XGBoost | `XGBRegressor(random_state=42)`, trained separately per commodity |
| AutoGluon Tabular | `presets="medium_quality"`, `time_limit=300`, `eval_metric="mean_absolute_error"`, trained separately per commodity |

AutoGluon internally evaluates multiple candidates (LightGBM, XGBoost, WeightedEnsemble, NeuralNetTorch, ExtraTrees, RandomForest, etc.) and selects the best model via internal validation score (`score_val`).

### Evaluation Metrics

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error, primary metric)
- **MAPE** (Mean Absolute Percentage Error)

---

## Experiments

### Notebook 1: Full-Weather Model
`Price_Prediction_Based_on_the_Weather_ML_C_9ipynb.ipynb`

- Trains XGBoost and AutoGluon with both price and weather lag features
- Performs lag window ablation across N = 1 to 7
- Reports per-commodity results for the optimal configuration (N = 2)

### Notebook 2: No-Weather Baseline
`Price_Prediction_Without_Weather_ML_C_9.ipynb`

- Trains XGBoost and AutoGluon using price lag features only
- Performs lag window ablation across N = 1 to 7
- Serves as the baseline to quantify the contribution of weather variables

---
