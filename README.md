# SignalGuard

**Physiological rPPG Quality Gating & Baseline Deepfake Classifier Pipeline**

SignalGuard is a forensic video-analysis pipeline designed to extract remote photoplethysmography (rPPG) signals from facial regions, evaluate the physiological reliability of the extracted blood volume pulse (BVP), and compare un-gated versus physiologically-gated deepfake detection.

---

## Architecture Overview

```
                               ┌─────────────────────────┐
                               │    Input Video (.mp4)   │
                               └────────────┬────────────┘
                                            │
                         ┌──────────────────┴──────────────────┐
                         │                                     │
                         ▼                                     ▼
           ┌───────────────────────────┐         ┌───────────────────────────┐
           │     SignalGuard Gate      │         │    Baseline Classifier    │
           │   (rPPG, POS, Forensics)  │         │ (2D FFT, Spatial Boundary)│
           └─────────────┬─────────────┘         └─────────────┬─────────────┘
                         │                                     │
                         │ Eligibility Verdict                 │ Calibrated P(manipulated)
                         ▼                                     ▼
             [ ELIGIBLE / INCONCLUSIVE ]               [ GENUINE / MANIPULATED ]
                         │                                     │
                         └──────────────────┬──────────────────┘
                                            ▼
                            Controlled Benchmark Evaluation
                         (Gated vs. Un-gated Classification)
```

---

## Key Modules

* **`signal_extractor.py`**: Tracks 468 facial landmarks via MediaPipe FaceMesh to isolate forehead and cheek regions of interest (ROIs), with automatic yaw rotation fallback.
* **`pos_extractor.py`**: Plane-Orthogonal-to-Skin (POS) rPPG extraction algorithm converting regional RGB signals into temporal pulse waveforms.
* **`signalguard_utils.py`**: Centralized DSP routines, zero-phase 4th-order Butterworth bandpass filtering (0.7–3.5 Hz / 42–210 BPM), Welch PSD spectral estimation, short-gap cubic interpolation, motion artifact SNR weighting, and regional agreement metrics.
* **`eligibility_gate.py`**: Quality gate evaluating temporal consistency, SNR thresholds, and forehead-vs-cheek agreement to issue an `ELIGIBLE` or `INCONCLUSIVE` verdict.
* **`baseline_classifier.py`**: Standalone deepfake detector extracting 2D-FFT high-frequency residuals, azimuthal spectral variance, edge gradient density, and boundary Laplacian metrics. Trained using L2-regularized logistic regression with Platt scaling probability calibration.
* **`dataset_loader.py`**: Ingests dataset manifests, enforces strict identity-grouped splitting (`split_by_subject`) to prevent data leakage, and provides an offline synthetic validation fixture generator.
* **`classifier_evaluator.py`**: Computes confusion matrices (TN, FP, FN, TP), precision, recall, specificity, FPR, FNR, balanced accuracy, F1 score, ROC-AUC, and per-condition performance breakdowns.
* **`stress_pipeline_runner.py`**: Stress tests pipeline resilience against brightness degradation, spatial downsampling, frame drops, and heavy compression.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/karanssable70-png/Signal-Guard.git
   cd Signal-Guard
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Generate Synthetic Validation Fixture
Generate a leak-proof synthetic test dataset partitioned into train, validation, and test splits:
```bash
python dataset_loader.py --generate-fixture fixtures/synthetic_dataset --num-subjects 8 --clips-per-subject 4 --split
```

### 2. Train the Baseline Classifier
Train the calibrated baseline classifier and save model weights and metadata:
```bash
python baseline_classifier.py --train-manifest fixtures/synthetic_dataset/manifest.csv --output-model models/baseline_classifier.joblib
```

### 3. Evaluate the Classifier
Run evaluation on the test split:
```bash
python classifier_evaluator.py --model models/baseline_classifier.joblib --manifest fixtures/synthetic_dataset/manifest.csv --split test --output-json results/baseline_test_eval.json
```

### 4. Run Standalone Inference on a Video
Predict manipulation probability on any video:
```bash
python baseline_classifier.py --predict "path/to/video.mp4"
```
Example Output:
```json
{
  "video": "path/to/video.mp4",
  "prediction": "GENUINE",
  "probability_manipulated": 0.0002,
  "probability_genuine": 0.9998,
  "confidence": 0.9998,
  "threshold": 0.5,
  "score_interpretation": "Calibrated posterior probability P(manipulated | video_features) in [0.0, 1.0]. Values >= 0.5 predict MANIPULATED.",
  "status": "SUCCESS",
  "error_message": null
}
```

### 5. Run Physiological Pipeline & Stress Tests
```bash
# Run unit tests
python -m unittest test_suite.py
python -m unittest test_baseline_classifier.py

# Run stress evaluations across degradations
python run_all_stress_eval.py
```

---

## Research Benchmark Datasets

The dataset loader is architected for **FaceForensics++ (FF++)**:
* **Formats**: Supports `c0` (raw), `c23` (light compression), and `c40` (heavy compression).
* **Partitioning**: Strictly isolates the 1,000 source subjects into 720 train / 140 val / 140 test splits to prevent identity leakage across splits.
