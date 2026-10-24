# OpenSense-HAR

Code for the manuscript "Decoupled Language Grounding and Uncertainty-Aware Fusion for Open-World Multimodal Human Activity Recognition" (OpenSense-HAR), evaluated on CZU-MHAD (depth video, 3-D skeleton and ten body-worn IMUs).

Everything sits in this one folder. Scripts create `cache/`, `results/` and `logs/` next to themselves when they run. `reference_summary.json` and `reference_bench.json` hold the numbers reported in the paper, for comparison with a fresh run.

## 1. Environment

- Python 3.10 or newer (the paper used 3.13)
- `pip install -r requirements.txt`

A GPU is not required. All reported runs used two CPU cores.

## 2. Data

CZU-MHAD is distributed by its authors (X. Chao, Z. Hou and Y. Mo, IEEE Sensors J., 2022, doi:10.1109/JSEN.2022.3150225). Each modality lives on its own branch of the dataset repository:

```bash
mkdir -p data && cd data
git clone --depth 1 --branch skeleton --single-branch https://github.com/yujmo/CZU_MHAD skel
git clone --depth 1 --branch sensors  --single-branch https://github.com/yujmo/CZU_MHAD sens
git clone --depth 1 --no-checkout --branch depth --single-branch https://github.com/yujmo/CZU_MHAD depth
cd ..
```

The depth branch is about 7.5 GB. `prep_depth.py` checks it out one subject at a time.

The GloVe vectors (glove-wiki-gigaword-300) come from the gensim-data release:

```bash
mkdir -p assets
curl -L -o assets/glove300.gz https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-300/glove-wiki-gigaword-300.gz
```

If your data is not in `./data`, set the paths at the top of `prep_raw.py` and `prep_depth.py`.

## 3. Pipeline

```bash
bash run_all.sh          # full pipeline
QUICK=1 bash run_all.sh  # 2-epoch smoke test on one fold
```

| Step | Script | Output |
|---|---|---|
| Parse skeleton and IMU text files | `prep_raw.py` | `cache/raw.pkl` |
| Align, resample, recover IMU placement | `prep_seq.py` | `cache/seq.npz` |
| Segment, back-project and render depth (0/30/60 degree yaw) | `prep_depth.py` | `cache/depth.npz` |
| Text features from the descriptions | `build_text.py`, `descriptions.py` | `cache/text_tok.npz` |
| Closed-set, open-world, limited-label and ablation runs | `run.py` (one job per call), `jobs.txt`, `worker.sh` | `results/<exp>/*.pkl` |
| Sequential deployment with test-time adaptation | `tta.py <fold> closed\|open` | `results/tta/` |
| Statistics and summary | `analysis.py` | `results/summary.json` |
| Latency and parameter count | `bench.py` | `results/bench.json` |
| Figures | `fig_arch.py`, `fig_mech.py`, `figs_results.py` (helpers in `diagram.py`) | PNG files in this folder |

## 4. Model code

| File | Contents |
|---|---|
| `models.py` | Primitive tokenisers, shared hierarchical temporal encoder, joint cross-modal expert, text encoder, aleatoric and MC-dropout reliability heads, uncertainty-gated latent generator, product-of-experts fusion, prototype classifier and decoupled semantic head; baselines (ST-GCN-style, DeepConvLSTM, depth TCN, early fusion, HAMLET-style attention fusion, XM-TF, ActionMAE-style, language binding) |
| `train.py` | Two-stage training (masked multimodal modelling, then language-grounded alignment) and inference with MC dropout |
| `data.py` | Loading, leave-one-subject-out and leave-activity-out splits, augmentation, controlled corruptions |
| `run.py` | Named configurations (`CONFIGS`) for the proposed model (`opensense`), baselines and ablations (`abl_*`) |
| `tta.py` | Sequential deployment with Tent, EATA, CoTTA and the confidence-filtered LoRA adapter plus prototype-memory method |

## 5. Reproducibility notes

- Seeds are fixed (seed 0) for every reported run.
- `worker.sh` uses EPOCHS=25, PRE_EPOCHS=10 and P_DROP=0.25. These are the settings behind the reported numbers.
- `skip.txt` lists queued jobs that were not run for the paper (second-seed repeats and one limited-label baseline).
- The cross-view conditions come from re-projecting the real depth point cloud and skeleton. They are not recordings from a second camera.

## 6. Licence

The code is released for research use. CZU-MHAD is subject to its authors' terms.
