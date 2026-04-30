"""Tests des services d'analyse — INF232 EC2"""
import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Dataset numérique simple."""
    np.random.seed(42)
    n = 80
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y  = 3 * x1 + 1.5 * x2 + np.random.randn(n) * 0.5
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def sample_classification_df():
    """Dataset de classification."""
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    labels = (x1 + x2 > 0).astype(int)
    label_str = ["A" if l == 0 else "B" for l in labels]
    return pd.DataFrame({"x1": x1, "x2": x2, "label": label_str})


@pytest.fixture
def sample_multiclass_df():
    """Dataset multi-classes."""
    np.random.seed(0)
    n = 90
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    classes = np.random.choice(["A", "B", "C"], size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "class": classes})


# ── Tests Régression ─────────────────────────────────────────────────────────

class TestRegressionService:
    def test_simple_regression_returns_keys(self, sample_df):
        from app.services.regression_service import simple_regression
        result = simple_regression(sample_df, "x1", "y")
        assert "r2" in result
        assert "mse" in result
        assert "rmse" in result
        assert "equation" in result
        assert "plot" in result

    def test_simple_regression_r2_positive(self, sample_df):
        from app.services.regression_service import simple_regression
        result = simple_regression(sample_df, "x1", "y")
        assert 0 <= result["r2"] <= 1

    def test_multiple_regression_returns_keys(self, sample_df):
        from app.services.regression_service import multiple_regression
        result = multiple_regression(sample_df, ["x1", "x2"], "y")
        assert "r2" in result
        assert "r2_adj" in result
        assert "coefficients" in result

    def test_multiple_regression_high_r2(self, sample_df):
        from app.services.regression_service import multiple_regression
        result = multiple_regression(sample_df, ["x1", "x2"], "y")
        assert result["r2"] > 0.8, f"R² trop bas : {result['r2']}"

    def test_simple_regression_error_on_few_rows(self):
        from app.services.regression_service import simple_regression
        tiny_df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        with pytest.raises(ValueError, match="Pas assez"):
            simple_regression(tiny_df, "x", "y")


# ── Tests Réduction ───────────────────────────────────────────────────────────

class TestReductionService:
    def test_pca_returns_variance(self, sample_df):
        from app.services.reduction_service import apply_pca
        result = apply_pca(sample_df, ["x1", "x2", "y"])
        assert "variance_ratio" in result
        assert "plot" in result
        assert len(result["variance_ratio"]) >= 1

    def test_pca_cumulative_near_100(self, sample_df):
        from app.services.reduction_service import apply_pca
        result = apply_pca(sample_df, ["x1", "x2", "y"])
        assert result["cumulative_variance"][-1] > 95

    def test_tsne_returns_plot(self, sample_df):
        from app.services.reduction_service import apply_tsne
        result = apply_tsne(sample_df, ["x1", "x2", "y"], n_iter=250)
        assert "plot" in result
        assert "kl_divergence" in result

    def test_lda_returns_classes(self, sample_classification_df):
        from app.services.reduction_service import apply_lda
        result = apply_lda(sample_classification_df, ["x1", "x2"], "label")
        assert "classes" in result
        assert len(result["classes"]) == 2


# ── Tests Classification supervisée ──────────────────────────────────────────

class TestSupervisedService:
    def test_knn_accuracy_range(self, sample_classification_df):
        from app.services.supervised_service import classify
        result = classify(sample_classification_df, ["x1", "x2"], "label", "knn")
        assert 0 <= result["accuracy"] <= 1

    def test_random_forest_feature_importances(self, sample_classification_df):
        from app.services.supervised_service import classify
        result = classify(sample_classification_df, ["x1", "x2"], "label", "random_forest")
        assert len(result["feature_importances"]) == 2

    def test_svm_returns_report(self, sample_classification_df):
        from app.services.supervised_service import classify
        result = classify(sample_classification_df, ["x1", "x2"], "label", "svm")
        assert "classification_report" in result
        assert "precision" in result

    def test_decision_tree_cv_scores(self, sample_classification_df):
        from app.services.supervised_service import classify
        result = classify(sample_classification_df, ["x1", "x2"], "label", "decision_tree")
        assert "cv_scores" in result
        assert len(result["cv_scores"]) == 5

    def test_multiclass_classification(self, sample_multiclass_df):
        from app.services.supervised_service import classify
        result = classify(sample_multiclass_df, ["x1", "x2"], "class", "random_forest")
        assert result["n_classes"] == 3


# ── Tests Classification non-supervisée ──────────────────────────────────────

class TestUnsupervisedService:
    def test_kmeans_returns_clusters(self, sample_df):
        from app.services.unsupervised_service import kmeans_clustering
        result = kmeans_clustering(sample_df, ["x1", "x2"], n_clusters=3)
        assert result["n_clusters"] == 3
        assert "silhouette_score" in result
        assert "cluster_sizes" in result

    def test_kmeans_cluster_sum_equals_n(self, sample_df):
        from app.services.unsupervised_service import kmeans_clustering
        result = kmeans_clustering(sample_df, ["x1", "x2"], n_clusters=3)
        total = sum(result["cluster_sizes"].values())
        assert total == len(sample_df)

    def test_dbscan_returns_noise(self, sample_df):
        from app.services.unsupervised_service import dbscan_clustering
        result = dbscan_clustering(sample_df, ["x1", "x2"], eps=0.5, min_samples=5)
        assert "n_noise" in result
        assert result["n_samples"] == len(sample_df)

    def test_hierarchical_ward(self, sample_df):
        from app.services.unsupervised_service import hierarchical_clustering
        result = hierarchical_clustering(sample_df, ["x1", "x2"], n_clusters=3, linkage_method="ward")
        assert result["n_clusters"] == 3
        assert "silhouette_score" in result
        assert "plot" in result

    def test_kmeans_silhouette_valid_range(self, sample_df):
        from app.services.unsupervised_service import kmeans_clustering
        result = kmeans_clustering(sample_df, ["x1", "x2"], n_clusters=2)
        assert -1 <= result["silhouette_score"] <= 1


# ── Tests Data Service ────────────────────────────────────────────────────────

class TestDataService:
    def test_get_dataset_info(self):
        from app.services.data_service import get_dataset_info
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.1, None, 3.3]})
        info = get_dataset_info(df)
        assert info["n_rows"] == 3
        assert info["n_cols"] == 3
        assert info["missing_values"] == 1
        assert "a" in info["numeric_columns"]

    def test_descriptive_stats(self):
        from app.services.data_service import get_descriptive_stats
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        stats = get_descriptive_stats(df)
        assert "x" in stats
        assert stats["x"]["mean"] == 3.0
        assert stats["x"]["min"] == 1.0

    def test_get_dataframe_preview(self):
        from app.services.data_service import get_dataframe_preview
        df = pd.DataFrame({"a": range(20), "b": range(20)})
        cols, rows = get_dataframe_preview(df, n=5)
        assert len(rows) == 5
        assert "a" in cols

    def test_allowed_file(self):
        from app.services.data_service import allowed_file
        assert allowed_file("data.csv")
        assert allowed_file("data.xlsx")
        assert not allowed_file("data.txt")
        assert not allowed_file("data.py")
