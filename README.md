# Monitoring module

Drift checks for model inputs and outputs.

Requires `numpy` and `pandas`.

---

## PredictorsMonitor

Compares a new predictors DataFrame against an etalon. Reports PSI, missing-value share, and (for categorical predictors) new/missing categories.

```python
from monitoring import PredictorsMonitor

predictors_monitor = PredictorsMonitor(bins_amt=10)
predictors_monitor.fit(etalon_predictors)
report = predictors_monitor.monitor(test_predictors)
```

| Method | Description |
|--------|-------------|
| `__init__(bins_amt=10)` | `bins_amt` sets bin count for numerical predictors. |
| `fit(data, checks=None)` | Fit on etalon predictors. `checks` overrides thresholds (`{pred}__PSI`, `{pred}__NA_PERC`). Defaults: `0.2` / `0.1`. |
| `get_test_stat(data)` | Stats for the test set aligned to the etalon. |
| `monitor(data)` | Full check; returns a summary `DataFrame`. |

Checks:

- **PSI** — distribution shift vs etalon
- **NA_PERC** — change in missing-value share
- **NEW_VAL** — new categories in test not present in etalon (categorical only)
- **NO_VAL** — categories from etalon missing in test (categorical only)

Output columns: `PRED_NAME`, `PRED_TYPE`, `CHECK_TYPE`, `CHECK_VALUE`, `CHECK_STATE` (`OK` / `NOT_OK`).

---

## PredictionsMonitor

Compares new prediction arrays (`np.ndarray`) against an etalon. Reports PSI, outlier share, and (for classification) mean entropy or (for regression) mean prediction.

```python
from monitoring import PredictionsMonitor

predictions_monitor = PredictionsMonitor(task="classification", bins_amt=10)
predictions_monitor.fit(etalon_predictions)
report = predictions_monitor.monitor(test_predictions)
```

| Method | Description |
|--------|-------------|
| `__init__(task="classification", bins_amt=10)` | `task` is `classification` or `regression`; `bins_amt` sets bin count. |
| `fit(data, checks=None)` | Fit on etalon predictions. `checks` overrides thresholds (`{PREFIX}_{i}__PSI`, `{PREFIX}_{i}__OUTL_PERC`). Defaults: `0.2` / `0.05`. |
| `get_test_stat(data)` | Stats for the test set aligned to the etalon. |
| `monitor(data)` | Full check; returns a summary `DataFrame`. |

Checks:

- **PSI** — distribution shift vs etalon
- **OUTL_PERC** — change in outlier share
- **MEAN_ENTROPY** — shift in mean prediction entropy (classification only)
- **MEAN** — shift in mean prediction (regression only)

Output columns: `{CLASS|REGRESSOR}_NAME`, `CHECK_TYPE`, `CHECK_VALUE`, `CHECK_STATE` (`OK` / `NOT_OK`).
