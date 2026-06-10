# ETA Regressor

This notebook contains the full development pipeline for the ATAS minimum evasion time regressor used in Stage 4 of the project.

---

## What the model actually predicts

This model does not predict raw time to impact.

Physics already handles time to impact directly from distance and closure rate. The ML model solves a different problem:

> How much time does the pilot actually need to react and begin evasive action given the current engagement state?

That includes non linear interactions between:

* Aircraft maneuverability
* Altitude
* Missile phase
* Aspect angle
* Countermeasure deployment
* G force constraints

These interactions become difficult to model cleanly with equations alone once sensor inputs become incomplete or noisy.

---

## Dataset generation

| Property           | Value                                |
| ------------------ | ------------------------------------ |
| Dataset type       | Synthetic                            |
| Generation method  | Physics based Monte Carlo simulation |
| Dataset size       | 1,000,000 rows                       |
| Label source       | Physics equations + controlled noise |
| Generator location | `src/physics_generator.py`           |

The dataset was generated synthetically because real missile engagement telemetry is not publicly available.

Instead of collecting imperfect real world data, the pipeline generates fully known engagement scenarios using physics equations. The ML model then learns the patterns across those scenarios and later generalizes from incomplete inputs during inference.

This ended up being one of the most important design decisions in the project.

---

## Feature set

### Engagement geometry

| Feature              | Description                             |
| -------------------- | --------------------------------------- |
| `launch_distance`    | Missile launch distance in metres       |
| `remaining_distance` | Current distance to impact in metres    |
| `closure_rate`       | Combined closing speed in m/s           |
| `aspect_angle`       | Incoming threat angle                   |
| `missile_phase`      | 0 = boost, 1 = mid course, 2 = terminal |
| `azimuth`            | Horizontal threat angle                 |
| `elevation`          | Vertical threat angle                   |

### Your aircraft state

| Feature                | Description                   |
| ---------------------- | ----------------------------- |
| `your_speed`           | Current airspeed              |
| `your_altitude`        | Current altitude              |
| `enemy_altitude`       | Enemy aircraft altitude       |
| `your_maneuverability` | 0 = low, 1 = medium, 2 = high |

### Threat aircraft specs

| Feature            | Description                     |
| ------------------ | ------------------------------- |
| `missile_speed`    | Incoming missile speed          |
| `missile_range`    | Maximum effective missile range |
| `enemy_generation` | Aircraft generation rating      |

### Countermeasure state

| Feature                   | Description                    |
| ------------------------- | ------------------------------ |
| `countermeasure_deployed` | 0 = not deployed, 1 = deployed |

The feature set is intentionally locked at 14 features.

---

## Model evolution

Three regressors were trained and compared during development.

### Random Forest Regressor

Used as the baseline model. It achieved reasonable R² performance but had slower inference and limited scaling during hyperparameter search.

### LightGBM Regressor

Trained faster than Random Forest and produced competitive metrics, but exposed fewer tunable parameters during optimization.

### XGBoost Regressor

XGBoost became the final choice for four reasons:

* Best MAE across all tested models
* Native GPU acceleration for Optuna tuning
* Larger hyperparameter search space
* SHAP explainability support

The final model architecture is:

| Component            | Configuration     |
| -------------------- | ----------------- |
| Model                | XGBoost Regressor |
| Objective            | Regression        |
| Tuning framework     | Optuna            |
| Serialization format | `.joblib`         |

---

## Optuna tuning

| Parameter           | Value              |
| ------------------- | ------------------ |
| Trials              | ~944               |
| Optimization target | Validation MAE     |
| Early stopping      | Monitored manually |

The search space included:

* `n_estimators`
* `max_depth`
* `learning_rate`
* `subsample`
* `colsample_bytree`
* `min_child_weight`
* `gamma`
* `reg_alpha`
* `reg_lambda`

Training stopped around 944 trials because validation MAE improvements became marginal beyond that point.

---

## Final metrics

| Metric         | Score          |
| -------------- | -------------- |
| R²             | 0.9939         |
| MAE            | 0.4552 seconds |
| RMSE           | 2.5089 seconds |
| Dangerous Rate | 0.128%         |

Dangerous Rate tracks how often the model underestimates minimum evasion time beyond a defined safety threshold.

This metric mattered more than RMSE during development because underestimating available reaction time is the failure mode that matters in a threat scenario.

---

## Inference rule

Predictions are always clipped at zero during inference:

```python id="3b5y7k"
np.maximum(model.predict(features), 0)
```

Negative evasion time has no physical meaning, so this rule is enforced at every inference call.

---

## Saved model

```text id="b98mw8"
release_models/
└── eta/
    └── atas_final_eta_regressor_model.joblib
```

| Property | Value     |
| -------- | --------- |
| Format   | `.joblib` |
| Storage  | GitHub    |

The final serialized model remains within GitHub storage limits and is versioned directly inside the repository.

---

## Key design decisions

* The model predicts minimum evasion time, not raw time to impact. Physics already solves impact timing directly.

* Synthetic physics generated data was used because large scale real engagement telemetry is not publicly available.

* Dangerous Rate was tracked separately because underestimating reaction time is the critical failure mode in a threat scenario.

* XGBoost outperformed Random Forest and LightGBM while also supporting GPU accelerated Optuna tuning and SHAP explainability.
