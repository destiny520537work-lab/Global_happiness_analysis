# Global Happiness Analysis

Analyze global happiness index data using R and Python to explore key factors affecting happiness.

## Project Structure

```
Global_happiness_analysis/
├── data/                  # Data files (add Happy_Updated.csv here)
├── R/
│   └── data_analysis_happiness_modeling.R   # R modeling script
├── app/
│   ├── app.py             # Streamlit interactive app
│   └── requirements.txt   # Python dependencies
├── figures/               # Generated charts after running
├── outputs/               # Generated result files after running
└── .github/workflows/     # GitHub Actions workflow
```

## Quick Start

### 1. Prepare Data

Place `Happy_Updated.csv` in the `data/` directory.

### 2. Run R Analysis Script

```r
# Install dependencies
install.packages(c("tidyverse", "here", "broom", "patchwork",
                   "glmnet", "caret", "MASS", "car", "lmtest", "corrplot"))

# Run script
source("R/data_analysis_happiness_modeling.R")
```

After running, check:
- `figures/` - Correlation matrix, residual plots, and other visualizations
- `outputs/` - Model evaluation results in CSV format

### 3. Run Streamlit Web App

```bash
cd app
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

Open your browser and visit http://localhost:8501

## Features

- **R Script**: Compare multiple models including OLS regression, Ridge, LASSO, and Robust regression
- **Web App**: Interactive data exploration, model parameter tuning, real-time visualization

## Dependencies

**R packages**: tidyverse, here, broom, patchwork, glmnet, caret, MASS, car, lmtest, corrplot

**Python packages**: streamlit, pandas, numpy, scipy, scikit-learn, matplotlib, seaborn
