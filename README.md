# Bank_Customer_Segmentation
This project builds a clean, leakage-free customer segmentation and profiling pipeline for bank transactions. It fixes flaws from naive pipelines (like IQR outlier dropping and unscaled categorical dominance) by using log-transformed features, K-Means, PCA visualization, and a non-circular segment classifier validated via cross-validation.
# Bank Customer Segmentation (K-Means + PCA) with a Leakage-Free Segment Classifier
 
Segments bank transactions into actionable customer groups with K-Means, then tests whether a segment can be predicted from information that was *not* used to build it (no circular evaluation).
 
## Key results
 
| Item | Result |
|---|---|
| Clean rows used | 2,321 of 2,537 (duplicates and missing values removed) |
| Segmentation | K-Means, k = 3, silhouette 0.23 (overlapping groups along age and balance) |
| Segment classifier (held-out features) | 82% accuracy vs. 59% majority baseline; macro-F1 0.54 vs. 0.25 (5-fold CV) |
| Login-anomaly prediction | ROC-AUC 0.52, i.e. not predictable from the available attributes |
 
**Segments (k = 3, n = 2,321 transactions):**
 
| Segment | Share | Avg age | Avg balance | Profile |
|---|---|---|---|---|
| A. Young, low balance | 5% | 29 | ~$445 | 79% students; transactions ~1.6x the balance |
| B. Young, mid balance | 36% | 29 | ~$2,285 | growth segment |
| C. Established, high balance | 59% | 56 | ~$7,160 | doctors, engineers, retirees; retention segment |
 
Occupation alone identifies segment C almost perfectly but cannot recover segment A, so balance has to be observed directly. Login-attempt anomalies (4.7% of rows) are not predictable from any available attribute, consistent with a synthetic dataset; this is reported as a negative result rather than forced into a model.
 
## Method
1. Audit: missing values (~1% per column), 21 duplicates, `LoginAttempts` distribution.
2. Features: log-transformed amount, amount-to-balance ratio, age, balance, duration; outliers kept (no IQR deletion); `Location` excluded (43 unordered categories); all features scaled.
3. K-Means with silhouette and inertia over k = 2-8; PCA for visualization.
4. Segment classifier on features *excluded* from clustering (occupation, channel, type, login attempts), 5-fold stratified CV against a majority-class baseline.
## Run
```
pip install pandas numpy scikit-learn matplotlib
jupyter notebook bank_customer_segmentation.ipynb
```
Data: `bank_transactions_data_edited.csv` (synthetic, course-provided).
 
## Limitations / next steps
Synthetic data; moderate separation; segments reflect age and balance more than behavior. Next: add recency/frequency/merchant features, bootstrap-validate cluster stability, compare Gaussian mixtures and hierarchical clustering.
