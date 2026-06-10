# Hit Classifier

This notebook contains the full development pipeline for the ATAS hit probability classifier used in the ATAS threat assessment pipeline

---

## What the model predicts

This model predicts survival probability after evasion.

This is fundamentally different from physics trajectory calculation. Physics can estimate where a missile is going in a straight line. It cannot reliably predict whether the missile still hits after the pilot maneuvers, changes altitude, and deploys countermeasures simultaneously.

That becomes a pattern recognition problem learned from engagement outcomes rather than a direct equation.

At inference time the model outputs:

* Binary prediction → hit or miss
* Probability score → used directly as the HUD threat gauge percentage

---

## Why this model exists alongside the regressor

The ETA regressor tells you when.

The hit classifier tells you if.

Neither model is useful alone:

* ETA without hit probability tells you when impact occurs but not whether evasion succeeds
* Hit probability without ETA tells you danger exists but not when the pilot needs to react

Together they produce the threat state used by the tactical decision layer.

---

## Dataset

| Property      | Value                            |
| ------------- | -------------------------------- |
| Dataset type  | Synthetic                        |
| Source        | Physics based scenario generator |
| Dataset size  | 1,000,000 rows                   |
| Label         | Binary classification            |
| Label meaning | `1 = hit`, `0 = miss`            |

The dataset uses the same synthetic engagement generator as the ETA regressor.

Labels are generated from physics simulation outcomes with controlled noise added to create more realistic engagement behavior.

---

## Features that make this different from physics alone

Three features are especially important because they model behavior physics equations cannot handle cleanly from incomplete sensor inputs.

| Feature                   | Why it matters                                                                    |
| ------------------------- | --------------------------------------------------------------------------------- |
| `aspect_angle`            | Determines how aggressively the missile must turn to re acquire after maneuvering |
| `your_maneuverability`    | Defines the evasion envelope available to the pilot                               |
| `countermeasure_deployed` | Simulates guidance disruption and tracking interference                           |

The full feature set remains identical to the ETA regressor because both models evaluate the same engagement state from different perspectives.

---

## Model selection

XGBoost Classifier was selected directly based on earlier regression experiments on the same synthetic feature space.

A separate large scale model comparison was not necessary because XGBoost had already demonstrated:

* Strong performance on structured engagement data
* Stable probability outputs
* Efficient GPU accelerated training
* Clean SHAP explainability support

The baseline classifier already achieved extremely strong metrics, so additional large scale tuning was not pursued.

| Component            | Configuration         |
| -------------------- | --------------------- |
| Model                | XGBoost Classifier    |
| Task                 | Binary classification |
| Serialization format | `.joblib`             |

---

## Class imbalance handling

The dataset distribution was approximately:

* 59% hit
* 41% miss

This is a mild imbalance rather than a severe one, so no aggressive balancing strategy was necessary.

Class balance was preserved using stratified train, validation, and test splits:

```python id="o0wqjc"
stratify=aircraft_df['hit'],
random_state=42
```

No additional imbalance correction methods were used.

---

## Final metrics

| Metric   | Score  |
| -------- | ------ |
| Recall   | 0.9966 |
| F1 Score | 0.9968 |
| ROC AUC  | 0.9999 |

Recall above 0.996 means the classifier almost never misses a genuine hit prediction.

ROC AUC near 1.0 also shows that hit and miss outcomes become highly separable once the full engagement state is known.

---

## Saved model

```text id="b7dx5e"
release_models/
└── hit/
    └── atas_final_hit_classifier_model.joblib
```

| Property | Value     |
| -------- | --------- |
| Format   | `.joblib` |
| Storage  | GitHub    |

The serialized model remains within GitHub storage limits and is versioned directly inside the repository.

---

## Key design decisions

* The classifier predicts post evasion survival outcome, not missile trajectory.

* Recall was prioritized because false negatives are significantly more dangerous than false positives in a threat scenario.

* The raw classifier probability is passed directly to the HUD threat gauge rather than being converted into a handcrafted score.

* XGBoost was reused because it had already demonstrated strong performance on the same synthetic feature space and engagement data distribution.

