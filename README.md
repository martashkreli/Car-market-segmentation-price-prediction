# Car Price Prediction & Market Segmentation

> Predicting car prices from vehicle specifications and identifying market segments using unsupervised clustering — a full EDA-to-modeling pipeline in R.

---

## Context

A car manufacturer wants to launch a new model targeting a specific market segment. They need two things:
1. **A predictive model** to estimate the price of a new car based on its specifications
2. **A clustering model** to identify which market segment (SUV, sedan, coupe, etc.) the car belongs to — using only technical features, not body type labels

## Dataset

205 car models with 26 features including dimensions, engine specs, fuel economy, and price. Data cleaning involved extracting a `carCompany` column from `CarName`, correcting manufacturer name typos, and fixing fuel system labels.

## Approach

### Task 1 — Price Prediction

**EDA highlights:**
- Engine size has the strongest positive correlation with price (0.87)
- Highway MPG has the strongest negative correlation (-0.79)
- Luxury brands (Jaguar, Porsche, BMW) cluster at the top of the price range
- Created a weighted fuel efficiency feature combining city and highway MPG

**Models compared:**
- Linear models (AIC/BIC stepwise selection)
- Penalized regression (Ridge, Lasso, Elastic Net)
- Non-linear models (KNN, Splines, GAM, tree-based methods)

Data was split into train/test sets with model selection via validation performance.

### Task 2 — Market Segmentation (Clustering)

Using only technical specifications (excluding model name and body type), we clustered cars to discover natural market segments:
- K-Means vs. Hierarchical clustering
- Optimal cluster count selected via WSS, Silhouette, and Gap Statistic
- Results visualized in reduced-dimension space and compared against actual body type labels

## Key Findings

- Car size (wheelbase, length, width, curb weight) and power (engine size, horsepower) are the strongest price drivers
- Features like car height, stroke, peak RPM, and compression ratio have weak predictive power
- Higher-priced cars tend to prioritize performance over fuel economy (efficiency-price correlation: -0.70)
- Natural clusters align well with real market segments, validating the use of technical specs for segmentation

## Tech Stack

`R` · `ggplot2` · `caret` · `glmnet` · `rpart` · `cluster`

## Files

- `Projectorg2finalz.Rmd` — Full R Markdown analysis (EDA, modeling, clustering)
- `CarPrices.csv` — Dataset

## Author

**Marta Shkreli**

BSc in Computer Science and Management — Luiss Guido Carli, Rome
