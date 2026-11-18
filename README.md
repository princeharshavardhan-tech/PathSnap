# PathSnap

PathSnap matches a noisy GPS trip to directed road segments. A FastAPI server scores nearby roads with a trained LightGBM model and decodes a connected route with Viterbi. A browser on a second laptop can select a trip, display its map and read the server-side scores.

## Run the demonstration

Use Python 3.12. From the extracted project directory, run the following in Windows PowerShell. The private project download includes the original 80 training and 10 student-test trips in `data/`. For a public GitHub checkout, copy the supplied `train` and `student_test` folders from inside the course archive's `Student folder` directly into `data`. The expected layout is `data/student_test/00000052/00000052.track`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

In a second terminal, start the interface. Laptop 2 needs only Python and the `ui` folder. Run the command from the directory containing that folder.

```powershell
py -m http.server 8501 --bind 127.0.0.1 --directory ui
```

Open `http://localhost:8501`. For a one-laptop demo, enter `http://127.0.0.1:8000` as the model server. For two laptops on the same LAN, enter the IPv4 address of Laptop 1 with port 8000, for example `http://192.168.1.20:8000`. Use **Connect**, select a trip, then **Match route**. Keep the first terminal visible for the request logs. The working map uses bundled Leaflet assets and road geometry supplied with the trip.

On Linux or macOS, create the environment with `python3.12 -m venv .venv`. Use `.venv/bin/python` in place of the Windows interpreter path. Start the UI with `python3 -m http.server 8501 --bind 127.0.0.1 --directory ui`.

## Official evaluator mode

On Laptop 1 set the following variables **only when the organisers release their ten trips**. `PATHSNAP_TRIPS` points to the folder whose children are the ten trip ID directories. Keep the private reference root on the server. The scorer accepts either `<id>/<id>.route` or `<id>.route` within that root.

```powershell
$env:PATHSNAP_MODE = "evaluator"
$env:PATHSNAP_TRIPS = "D:\PathSnapEval\trips"
$env:PATHSNAP_REFERENCES = "D:\PathSnapEval\references"
.\.venv\Scripts\python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

The server lists available IDs dynamically. Prediction reads the track and road graph. Evaluation opens a reference after prediction and returns aggregate scores. Without a reference, the UI still displays the predicted route and its continuity. Each API worker loads the exported model once at startup.

| API | Result |
| --- | --- |
| `GET /health` | Service status, selected model and active split |
| `GET /trips` | The ten available trip IDs |
| `POST /match/{trip_id}` | GPS and road layers, inferred route, arc per observation, section boundaries, latency and available scores |
| `GET /docs` | FastAPI's interactive API reference |

## Reproduce the experiments

With the environment active, use the project root as the working directory. Feature caches are disposable. The raw files are read-only inputs.

```bash
python -m pathsnap.prepare --data data --cache cache --workers 2
python -m pathsnap.train --data data --cache cache
python -m pathsnap.visualize --data data
python -m pathsnap.verify --api http://127.0.0.1:8000 --offline outputs/predictions
```

The last command expects a running student-test server. `PathSnap.ipynb` presents the same workflow step by step. Each full training run recomputes five baselines. `--only LightGBM` trains an individual baseline. `--reuse-baselines` explicitly resumes an interrupted run using results from the same configuration and input data.

## Measured results

The 80 training trips were split into three whole-trip folds. Each fit samples up to 260 observations per training trip, and each fold scores up to 450 per held-out trip. All eight candidate rows for a sampled observation stay together. The ten student-test trips were scored in full, across 19,160 observations. Course route files contain an ordered sequence of traversed arcs. Pointwise agreement uses the monotone alignment documented in the report.

| Model | Mean grouped CV candidate macro F1 | Student-test Top-1 | Student-test candidate macro F1 |
| --- | ---: | ---: | ---: |
| XGBoost | 0.9337 | 88.60% | 0.9336 |
| LightGBM | 0.9341 | 88.73% | 0.9337 |
| Random Forest | 0.9335 | 88.78% | 0.9349 |
| KNN | 0.9256 | 87.93% | 0.9273 |
| HMM observation scorer | 0.8916 | 86.37% | 0.8911 |

The fifth model learns match-versus-background feature likelihoods and a distance-based transition model. Its standalone Viterbi result was **88.98% Top-1**. The final model was selected using grouped CV. Eight Bayesian-search configurations were recorded, including the initial setup, which retained the best score. Final LightGBM plus Viterbi reached **91.31% Top-1**, **98.11% strict continuity** and **0.9519 ordered-route F1** on student testing.

## Project files

| Location | Purpose |
| --- | --- |
| `pathsnap/data.py`, `reference.py`, `prepare.py` | Raw trip inputs, exact candidate search, separate reference alignment and dataset audit |
| `pathsnap/models.py`, `train.py` | Five scorers, grouped CV, Bayesian search, model selection and experiment exports |
| `pathsnap/decoder.py` | Directed network transitions, bounded shortest paths, Viterbi and ordered sequence metric |
| `models/` | Final native model and inference configuration |
| `pathsnap/api.py`, `service.py`, `ui/` | API, inference pipeline and evaluator interface |
| `outputs/` | Fold assignments, inventory, five baseline logs, search trials, predictions, charts and live API checks |
| `pathsnap/visualize.py` | Folium HTML with four toggleable layers and print figures. Run locally to regenerate the interactive map |
| `PathSnap_Technical_Report.pdf` | Consolidated report, metrics, screenshots, limitations and experiment details |

The classroom two-laptop demonstration and the hidden evaluator split require the organisers' machines and data. The included evidence was produced by two local services using the supplied student-test split. This implementation, experiment execution and report were prepared with AI assistance. Review the code and rerun the experiments as part of your team's independent evaluation.
