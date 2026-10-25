# Data

## `raw/synthetic_agritech_sensor_dataset.csv`

A **synthetic** multivariate time series that simulates sensor readings from an agricultural
processing line (for example grain conditioning and rolling). It holds 1,200 hourly records
from 2025-01-01 00:00 to 2025-02-19 23:00 and has no missing values. The pipeline still runs
KNN imputation, so it can also be applied to real sensor logs with gaps.

The data is synthetic so the method can be shared and reproduced without exposing
proprietary plant data.

### Variable dictionary

| Column | Unit | Role | Description | Range (min – max) |
|---|---|---|---|---|
| `timestamp` | datetime | index | Hourly acquisition time (dropped before modelling; row order keeps time order) | 2025-01-01 → 2025-02-19 |
| `temperature_c` | °C | environmental | Ambient temperature | 19.1 – 32.5 |
| `humidity_pct` | % RH | environmental | Relative humidity | 47.9 – 75.3 |
| `co2_ppm` | ppm | environmental | CO₂ concentration | 379 – 478 |
| `vibration_mm_s` | mm/s | machine state | Equipment vibration velocity | 1.3 – 2.3 |
| `throughput_setting` | % | **controllable** | Line throughput set point | 59.2 – 80.9 |
| `roller_pressure_kpa` | kPa | **controllable** | Roller / press pressure | 102.1 – 129.8 |
| `feed_moisture_pct` | % | process input | Moisture content of incoming feed | 11.1 – 16.8 |
| `energy_index` | index | **target 1 (minimize)** | Normalized energy consumption | 34.5 – 46.0 |
| `yield_efficiency_index` | index | **target 2 (maximize)** | Normalized yield / processing efficiency | 88.7 – 95.6 |

### Engineered features (built in `src/preprocessing.py`)

For each of the 7 non-target sensor variables:

- `<var>_lag1`: value at the previous time step
- `<var>_roll3`: 3-step rolling mean

This gives 21 candidate predictors. The top 10 by combined Random Forest importance become
the **control variables** that the optimizer searches over:

```
throughput_setting, throughput_setting_roll3, roller_pressure_kpa, roller_pressure_kpa_roll3,
throughput_setting_lag1, feed_moisture_pct_roll3, feed_moisture_pct, roller_pressure_kpa_lag1,
feed_moisture_pct_lag1, vibration_mm_s
```

Each control variable's search bounds are its observed min/max in the data, so the optimizer
does not extrapolate far outside the operating envelope.
