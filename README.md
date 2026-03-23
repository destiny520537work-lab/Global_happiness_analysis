# Global Happiness Analysis

**Exploring Global Patterns in National Happiness: A Multivariate Regression Analysis**

An end-to-end data science project analyzing the social, economic, health, and institutional determinants of national happiness across 146 countries.

---

## What This Project Does

Using the World Happiness Report dataset, this project:

- Builds and compares **7 regression models** (OLS, Ridge, LASSO, Robust, Stepwise, interaction terms, continental fixed effects) in R
- Provides an **interactive Streamlit web app** for real-time data exploration, model tuning, and visualization
- Automatically generates **model evaluation reports** (RMSE, MAE, R²) and **Chinese-language summaries** for each model
- Produces publication-quality figures: correlation matrix, LASSO regularization paths, residual diagnostics, and model comparison charts

## Key Findings (from the Paper)

| Predictor | Association with Happiness | +1 point in Ladder Score requires |
|-----------|--------------------------|-----------------------------------|
| Social Support | Strongest positive | +0.42 in Support |
| Freedom | Strong positive | +0.68 in Freedom |
| Corruption (perception) | Negative | +0.83 reduction |
| Healthy Life Expectancy | Positive | +30.5 years |
| Log GDP per capita | Positive | +2.21 in LGDP |

The final model (OLS with continental fixed effects) achieves **adj. R² ≈ 0.86**, with all five predictors statistically significant at the 5% level. North and South America show persistently higher happiness even after controlling for socioeconomic factors, suggesting regional effects beyond the measured variables.

> Social support is the single most influential predictor of national happiness — outpacing even economic prosperity.

## Project Structure

```
Global_happiness_analysis/
├── data/
│   └── Happy_Updated.csv          # Dataset (146 countries, 8 variables)
├── R/
│   └── data_analysis_happiness_modeling.R   # 7-model comparison pipeline
├── app/
│   ├── app.py                     # Streamlit interactive web app
│   └── requirements.txt           # Python dependencies
├── figures/                       # Auto-generated charts (after running R script)
│   ├── correlation_matrix.png
│   ├── lasso_path.png
│   └── model_rmse_compare.png
├── outputs/                       # Auto-generated result files (after running R script)
│   ├── model_evaluation_testset.csv
│   ├── coefficients_*.csv
│   └── automated_model_summaries_cn.txt
└── .github/workflows/             # GitHub Actions CI
```

## Quick Start

### 1. Run the R Analysis

```r
# Install dependencies (one-time)
install.packages(c("tidyverse", "here", "broom", "patchwork",
                   "glmnet", "caret", "MASS", "car", "lmtest", "corrplot"))

# Run the full pipeline from the project root
source("R/data_analysis_happiness_modeling.R")
```

This trains all 7 models, saves evaluation metrics to `outputs/`, and generates all figures to `figures/`.

### 2. Launch the Interactive Web App

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

The app lets you:
- Explore the dataset interactively (filter, sort, view distributions)
- Select predictors and model type (OLS / Ridge / LASSO)
- Tune regularization strength with a live slider
- Inspect fitted vs. actual plots and residual diagnostics

## Models Compared

| Model | Description |
|-------|-------------|
| `lm_base` | OLS with 5 continuous predictors |
| `lm_cont` | OLS + Continent fixed effects |
| `lm_inter` | OLS with interaction terms (LGDP×Support, HLE×Freedom) |
| `step_model` | Stepwise AIC selection from `lm_cont` |
| `rlm_model` | Robust regression (M-estimation, MASS::rlm) |
| `lasso` | LASSO with 5-fold CV lambda selection |
| `ridge` | Ridge with 5-fold CV lambda selection |

The best model (lowest test-set RMSE) is selected automatically and used for residual diagnostics.

## Dataset

`Happy_Updated.csv` — 146 countries, sourced from the World Happiness Report.

| Variable | Description |
|----------|-------------|
| `Ladder_score` | National happiness score (0–10, response variable) |
| `LGDP` | Log GDP per capita |
| `Support` | Perceived social support |
| `HLE` | Healthy life expectancy (years) |
| `Freedom` | Satisfaction with freedom of choice |
| `Corruption` | Perceived corruption in government/business |
| `Continent` | Categorical regional variable |

## Dependencies

**R:** `tidyverse`, `here`, `broom`, `patchwork`, `glmnet`, `caret`, `MASS`, `car`, `lmtest`, `corrplot`

**Python:** `streamlit`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`

## Paper

> *Exploring Global Patterns in National Happiness: A Multivariate Regression Analysis.*
> Group 14. Introduction to Statistics for Data Science (MATH42715), Durham University, 2025.
