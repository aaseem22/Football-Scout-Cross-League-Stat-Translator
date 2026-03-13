# ⚽ Football Scout — Cross-League Stat Translator

> **Can a neural network tell you how Kylian Mbappé would perform in the Premier League?**
> This project builds a CycleGAN that translates football player statistics between Ligue 1 and the Premier League — no paired transfer data required.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 🧠 The Problem I Was Solving

A scout watching Ligue 1 can't directly compare stats to the Premier League. A player scoring 0.5 goals/90 in Ligue 1 **is not the same** as 0.5 goals/90 in the Premier League — league difficulty, tempo, pressing intensity and tactical systems all compress or inflate numbers differently.

Traditional approaches use hand-crafted adjustment factors or regression models requiring labelled cross-league transfer data (player X moved from L1 to PL, here's what changed). That data is sparse and biased toward successful transfers.

**My approach:** use a CycleGAN to learn the full statistical distribution of each league and translate between them — the same way CycleGAN converts photos of horses to zebras, without needing paired examples.

---

## 🏗️ Architecture

```
Ligue 1 Stats (A) ──► G_AB ──► Fake PL Stats ──► D_B ──► Real or Fake?
                                      │
                               G_BA ◄─┘ (cycle consistency)
                                │
Premier League Stats (B) ──► G_BA ──► Fake L1 Stats ──► D_A ──► Real or Fake?
```

### Networks

| Component | Role | Architecture |
|---|---|---|
| **G_AB** | Ligue 1 → Premier League | 4-block MLP with LayerNorm + LeakyReLU |
| **G_BA** | Premier League → Ligue 1 | 4-block MLP with LayerNorm + LeakyReLU |
| **D_A** | Judges Ligue 1 samples | 3-block MLP with Dropout(0.3) |
| **D_B** | Judges PL samples | 3-block MLP with Dropout(0.3) |

### Loss Functions

```
Total Loss = L_adv + λ_cycle × L_cycle + λ_id × L_identity
```

- **Adversarial loss** — BCEWithLogitsLoss, generator fools discriminator
- **Cycle-consistency loss** — L1, λ=10, ensures A→B→A ≈ A (no information destroyed)
- **Identity loss** — L1, λ=5, G_AB(B) ≈ B (preserves stats already in target league)

### Training Config

```python
EPOCHS      = 300
BATCH_SIZE  = 32
LR          = 2e-4
BETAS       = (0.5, 0.999)   # Adam — standard GAN setting
LR_DECAY    = linear after epoch 150
HIDDEN_DIM  = 256
INPUT_DIM   = 13              # stat features per player
```

---

## 📊 Data

**Source:** Kaggle — European Football Player Stats (2017–2024)
**Leagues used:** Ligue 1, Premier League
**Filter:** ≥ 5 matches played per season

| Split | Players | Seasons |
|---|---|---|
| Ligue 1 | ~3,200 | 2017–18 to 2023–24 |
| Premier League | ~3,200 | 2017–18 to 2023–24 |

### 13 Features Translated

| Category | Features |
|---|---|
| Attacking | Goals/90, Assists/90, Shots/90, Shot accuracy %, Shot-creating/90, Goal-creating/90 |
| Possession | Pass completion %, Progressive passes, Progressive carries, Key passes, Dribble success % |
| Defensive | Interceptions, Tackles won |

---

## 🔍 What the Model Learned

After training, the CycleGAN captured real patterns that align with football knowledge:

- **Goals/90 compresses** — typically drops 0.03–0.10 moving L1→PL, reflecting higher defensive quality
- **Defensive stats rise** — tackles won and interceptions increase; the PL demands more pressing engagement
- **Pass completion falls** — ~0.5–2% drop; space closes faster under PL press
- **Dribble profiles shift** — progressive carries change significantly for wide forwards; position-specific patterns emerge cleanly

These aren't hand-coded adjustments — the model discovered them from distribution differences alone.

---

## 🖥️ The App

Built a full **Streamlit** scouting tool with 3 workflows:

### Tab 1 — Player Lookup
Search any player by name → auto-fills their stats from the dataset → CycleGAN translates instantly. Shows KPI cards with original vs projected values, a normalised radar chart, and a delta bar chart.

### Tab 2 — Head-to-Head Comparison
Compare two players' projected profiles side-by-side on an overlaid radar chart with a full head-to-head stat table.

### Tab 3 — Shortlist Ranker
Build a pool of targets, rank by any of 13 projected stats, visualise with grouped bar charts, and export rankings as CSV.

```bash
# Run locally
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
football_scout/
├── app.py                    # Streamlit application (4 tabs)
├── requirements.txt
├── models/
│   ├── G_AB.pt               # Trained generator: Ligue 1 → PL
│   ├── G_BA.pt               # Trained generator: PL → Ligue 1
│   └── scaler.pkl            # StandardScaler fitted on combined data
├── data/
│   ├── cleaned_2017-18.csv
│   ├── cleaned_2018-19.csv
│   ├── cleaned_2019-20.csv
│   ├── cleaned_2020-21.csv
│   ├── cleaned_2021-22.csv
│   ├── cleaned_2022-23.csv
│   └── cleaned_2023-24.csv
└── training/
    └── train_cyclegan.ipynb  # Full training pipeline (Google Colab)
```

---

## ⚙️ Setup & Usage

### Requirements

```
torch>=2.0
scikit-learn
joblib
streamlit
pandas
numpy
plotly
scipy
matplotlib
seaborn
```

### Run the app

```bash
git clone https://github.com/yourusername/football-scout
cd football-scout
pip install -r requirements.txt
streamlit run app.py
```

### Retrain the model

Open `training/train_cyclegan.ipynb` in Google Colab. The notebook handles:
1. Data loading and preprocessing
2. Model training with loss curves
3. KS-test evaluation of distribution alignment
4. PCA visualisation before/after translation
5. Model export (`G_AB.pt`, `G_BA.pt`, `scaler.pkl`)

---

## 📈 Evaluation

Beyond training loss, I evaluated the model using:

- **KS-test** (Kolmogorov-Smirnov) — measures whether the translated L1 distribution is statistically indistinguishable from real PL data. Green = p > 0.05 (distributions match), Red = still differs.
- **PCA alignment** — 2D scatter of real PL vs translated L1 shows how well the translated cloud overlaps the target distribution.
- **Visual stat distributions** — histograms of each feature before and after translation, compared against real PL data.

---

## 🚧 Limitations & Future Work

**Current limitations:**
- Goalkeepers have a structurally different stat profile — projections are unreliable for GKs
- Minimum 5 matches required — small-sample players are excluded
- Projects statistical output, not ability — system fit, injuries, motivation not captured
- Trained on Ligue 1 ↔ PL only; other league pairs would need retraining

**Future directions:**
- Extend to all 5 top European leagues
- Add position-conditional translation (train separate models per position group)
- Incorporate age curves to project stats forward in time
- Deploy to Streamlit Cloud with public access

---

## 💡 Key Takeaways

This project demonstrates:

- **Applying GANs beyond images** — CycleGAN on tabular sports data is underexplored and works well
- **Domain adaptation without paired data** — cycle-consistency is a powerful constraint when you can't get matched examples
- **End-to-end ML product** — from raw CSVs to a deployed scouting tool with a real UX
- **Interpretable ML** — the model outputs aren't black-box; every stat change is visible and explainable to a domain expert

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Model | PyTorch, scikit-learn |
| Training | Google Colab (GPU) |
| App | Streamlit, Plotly |
| Data | Pandas, NumPy, SciPy |
| Presentation | PptxGenJS |

---

## 📬 Contact

Built by **[Aaseem Mhaskar]**


- 💼 [LinkedIn](https://www.linkedin.com/in/aaseem-mhaskar-007279203)
- 🐙 [GitHub](https://github.com/aaseem22)

---

*If you work in football analytics or ML and want to chat about this project, feel free to reach out.*
