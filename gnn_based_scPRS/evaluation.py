import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer


# ------------------------------------------------
# Load PRS files
# ------------------------------------------------

def load_prs_data():

    cell_prs = pd.read_csv("celltype_prs_scores.csv")
    gene_prs = pd.read_csv("gene_level_prs.csv")

    print("\nCell-type PRS")
    print(cell_prs.head())

    print("\nGene PRS")
    print(gene_prs.head())

    return cell_prs, gene_prs


# ------------------------------------------------
# Build feature matrix
# ------------------------------------------------

def build_feature_matrix(cell_prs, gene_prs):

    # cell PRS features
    cell_features = cell_prs.set_index("cluster")["gene_prs"].to_frame().T
    cell_features.columns = ["PRS_" + c.replace(" ", "_") for c in cell_features.columns]

    # gene PRS features
    gene_features = gene_prs.groupby("gene")["gene_prs"].mean().to_frame().T
    gene_features.columns = ["PRS_" + g for g in gene_features.columns]

    cell_features = cell_features.reset_index(drop=True)
    gene_features = gene_features.reset_index(drop=True)

    X = pd.concat([cell_features, gene_features], axis=1)

    X = X.fillna(0)

    print("\nFeature matrix shape:", X.shape)

    return X


# ------------------------------------------------
# Simulate patients
# ------------------------------------------------

def simulate_patients(X, n_patients=200):

    print("\nSimulating patient cohort...")

    np.random.seed(42)   # reproducibility

    base_profile = X.iloc[0:1]

    X_sim = pd.concat([base_profile] * n_patients, ignore_index=True)

    noise = np.random.normal(0, 0.05, X_sim.shape)

    X_sim = X_sim + noise

    return X_sim


# ------------------------------------------------
# Simulate clinical outcomes
# ------------------------------------------------

def simulate_outcomes(n_patients):

    print("\nSimulating clinical outcomes...")

    np.random.seed(42)

    y = np.random.binomial(1, 0.5, size=n_patients)

    print("Cases:", sum(y))
    print("Controls:", len(y) - sum(y))

    return y


# ------------------------------------------------
# Preprocess features
# ------------------------------------------------

def preprocess_features(X):

    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


# ------------------------------------------------
# Evaluate models
# ------------------------------------------------

def evaluate_models(X_scaled, y):

    print("\nModel Evaluation\n")

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # L1 (LASSO)
    model_l1 = LogisticRegression(
        solver="liblinear",
        max_iter=2000,
        penalty="l1"
    )

    scores = cross_val_score(model_l1, X_scaled, y, cv=kfold, scoring="roc_auc")
    print("L1 (LASSO) AUC:", scores.mean())

    # L2 (Ridge)
    model_l2 = LogisticRegression(
        solver="lbfgs",
        max_iter=2000
    )

    scores = cross_val_score(model_l2, X_scaled, y, cv=kfold, scoring="roc_auc")
    print("L2 (Ridge) AUC:", scores.mean())

    # ElasticNet
    model_en = LogisticRegression(
        solver="saga",
        l1_ratio=0.5,
        max_iter=2000
    )

    scores = cross_val_score(model_en, X_scaled, y, cv=kfold, scoring="roc_auc")
    print("ElasticNet AUC:", scores.mean())


# ------------------------------------------------
# Feature importance
# ------------------------------------------------

def compute_feature_importance(X_scaled, y, feature_names):

    print("\nComputing feature importance...")

    model = LogisticRegression(
        solver="liblinear",
        max_iter=2000
    )

    model.fit(X_scaled, y)

    importance = pd.DataFrame({
        "feature": feature_names,
        "coef": model.coef_[0]
    })

    importance = importance.sort_values(
        "coef",
        key=abs,
        ascending=False
    )

    print("\nTop predictive PRS features:")
    print(importance.head(10))


# ------------------------------------------------
# Save dataset
# ------------------------------------------------

def save_dataset(X_sim, y):

    dataset = X_sim.copy()
    dataset["clinical_outcome"] = y

    dataset.to_csv("simulated_clinical_dataset.csv", index=False)

    print("\nSaved dataset: simulated_clinical_dataset.csv")


# ------------------------------------------------
# Main
# ------------------------------------------------

def main():

    cell_prs, gene_prs = load_prs_data()

    X = build_feature_matrix(cell_prs, gene_prs)

    X_sim = simulate_patients(X, 200)

    y = simulate_outcomes(200)

    X_scaled = preprocess_features(X_sim)

    print("\nDataset size check")
    print("X samples:", X_scaled.shape[0])
    print("y samples:", len(y))

    evaluate_models(X_scaled, y)

    compute_feature_importance(X_scaled, y, X_sim.columns)

    save_dataset(X_sim, y)


if __name__ == "__main__":
    main()