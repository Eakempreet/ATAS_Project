# App Backend

This folder contains the FastAPI backend that connects the HUD frontend to the ATAS inference pipeline.

---

## File overview

| File               | Purpose                                                                       |
| ------------------ | ----------------------------------------------------------------------------- |
| `main.py`          | FastAPI server that runs the inference pipeline and exposes the API endpoints |
| `test_pipeline.py` | End to end backend verification script                                        |

---

## main.py

`main.py` connects all `src/` modules together and exposes them through FastAPI.

The backend handles:

* Image upload
* Aircraft classification
* Metadata lookup
* Feature generation
* ML inference
* Tactical recommendations

---

### GET /

```http id="xxpr4l"
GET /
```

Returns:

```text id="n1b2qo"
frontend/atas_hud_v11.html
```

This serves the cockpit HUD interface directly from the backend.

---

### POST /analyze

```http id="0dcm0n"
POST /analyze
```

Accepts:

* Aircraft image upload
* Friendly aircraft selection
* Basic pilot state inputs

Current HUD controls:

* Your speed
* Your altitude

The remaining engagement parameters are currently generated internally to simulate live combat scenarios.

Future versions may expose more engagement controls directly through the HUD.

Pipeline flow:

```text id="xv93hk"
Image Upload
      ↓
Aircraft Classification
      ↓
Metadata Lookup
      ↓
Feature Generation
      ↓
ETA + Hit Prediction
      ↓
Tactical Recommendation
      ↓
JSON Response
```

The final response contains:

* Predicted aircraft
* Missile specifications
* ETA prediction
* Hit probability
* Tactical recommendation

---

### Model loading at startup

All models load once during server startup through the imported `src/` modules.

This keeps inference fast and avoids repeated disk loading during requests.

---

### Why `--reload` is avoided

Reload mode starts an extra FastAPI process and loads the models twice.

That wastes memory and can crash smaller GPUs.

---

## Test pipeline

`test_pipeline.py` verifies the backend before connecting the HUD frontend.

The script:

* Sends a real POST request
* Uses a real aircraft image
* Passes engagement parameters
* Checks the returned JSON response

---

### How to run the test

Start the server first:

```bash id="1fj3nm"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Run the test script:

```bash id="pl0z4j"
python -m app.test_pipeline
```

---

### Passing result

A successful run returns:

* HTTP status `200`
* Aircraft prediction
* ETA prediction
* Hit probability
* Tactical recommendation

Example:

```json id="u2u0vk"
{
  "aircraft_name": "Su57",
  "eta_seconds": 3.77,
  "hit_probability": 0.969,
  "recommendation": "BREAK HARD + DEPLOY CM + DISENGAGE IMMEDIATELY"
}
```

---

## How to run the server

### Local development

```bash id="6e3d6l"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:

```text id="j3cz51"
http://localhost:8000
```

---

### Docker production

```bash id="v7jmlf"
docker build -t atas .
```

```bash id="s9g4mh"
docker run -p 7860:7860 atas
```

---

## API reference

### GET /

| Property | Value                   |
| -------- | ----------------------- |
| Returns  | `atas_hud_v11.html`     |
| Purpose  | Serves the HUD frontend |

---

### POST /analyze

| Property     | Value                            |
| ------------ | -------------------------------- |
| Content type | `multipart/form-data`            |
| Purpose      | Runs the ATAS inference pipeline |

### Input fields

| Field               | Type       | Source              |
| ------------------- | ---------- | ------------------- |
| `image`             | UploadFile | Browser file upload |
| `your_speed`        | float      | HUD slider          |
| `your_altitude`     | float      | HUD slider          |
| `friendly_aircraft` | str        | HUD dropdown        |

### Internally generated engagement parameters

| Parameter                 |
| ------------------------- |
| `enemy_altitude`          |
| `countermeasure_deployed` |
| `launch_distance`         |
| `remaining_distance`      |
| `azimuth`                 |
| `elevation`               |

### Response JSON

| Field              | Type  |
| ------------------ | ----- |
| `aircraft_name`    | str   |
| `missile_speed`    | float |
| `missile_range`    | float |
| `enemy_generation` | float |
| `maneuverability`  | int   |
| `no_aa_capability` | int   |
| `eta_seconds`      | float |
| `hit_probability`  | float |
| `recommendation`   | str   |

---

## Key constraints

| Constraint                    | Reason                                     |
| ----------------------------- | ------------------------------------------ |
| Models load once at startup   | Keeps inference fast and stable            |
| `--reload` avoided            | Prevents duplicate model loading           |
| All imports use `src.` prefix | Keeps paths consistent across environments |
| Port `7860` used in Docker    | Required for Hugging Face Spaces           |
