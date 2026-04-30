from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
from app import db
from app.models.analysis import Analysis
from app.services.data_service import load_current_dataframe, get_dataset_info, get_current_dataset_filename
from app.services.unsupervised_service import kmeans_clustering, dbscan_clustering, hierarchical_clustering

unsupervised_bp = Blueprint("unsupervised", __name__, url_prefix="/analysis/unsupervised")


def _save_analysis(result, params):
    from app.models.dataset import Dataset
    filename = get_current_dataset_filename()
    ds = Dataset.query.filter_by(filename=filename, user_id=current_user.id).first()
    if ds:
        a = Analysis(user_id=current_user.id, dataset_id=ds.id,
                     analysis_type=f"unsupervised_{result['method'].lower().replace('-','_')}")
        a.set_params(params)
        a.set_results(result)
        db.session.add(a)
        db.session.commit()


@unsupervised_bp.route("/kmeans", methods=["GET", "POST"])
@login_required
def kmeans():
    df = load_current_dataframe()
    if df is None:
        flash("Aucun dataset chargé.", "warning")
        return redirect(url_for("data.upload"))

    info = get_dataset_info(df)
    numeric_cols = info.get("numeric_columns", [])
    result = None
    error = None

    if request.method == "POST":
        feature_cols = request.form.getlist("feature_cols")
        n_clusters = int(request.form.get("n_clusters", 3))

        if len(feature_cols) < 1:
            error = "Sélectionnez au moins 1 variable."
        else:
            try:
                result = kmeans_clustering(df, feature_cols, n_clusters)
                _save_analysis(result, {"features": feature_cols, "n_clusters": n_clusters})
                flash("K-Means effectué avec succès.", "success")
            except Exception as e:
                error = str(e)

    return render_template("analysis/unsupervised.html",
                           method="kmeans", numeric_cols=numeric_cols,
                           result=result, error=error)


@unsupervised_bp.route("/dbscan", methods=["GET", "POST"])
@login_required
def dbscan():
    df = load_current_dataframe()
    if df is None:
        flash("Aucun dataset chargé.", "warning")
        return redirect(url_for("data.upload"))

    info = get_dataset_info(df)
    numeric_cols = info.get("numeric_columns", [])
    result = None
    error = None

    if request.method == "POST":
        feature_cols = request.form.getlist("feature_cols")
        eps = float(request.form.get("eps", 0.5))
        min_samples = int(request.form.get("min_samples", 5))

        if len(feature_cols) < 1:
            error = "Sélectionnez au moins 1 variable."
        else:
            try:
                result = dbscan_clustering(df, feature_cols, eps, min_samples)
                _save_analysis(result, {"features": feature_cols, "eps": eps, "min_samples": min_samples})
                flash("DBSCAN effectué avec succès.", "success")
            except Exception as e:
                error = str(e)

    return render_template("analysis/unsupervised.html",
                           method="dbscan", numeric_cols=numeric_cols,
                           result=result, error=error)


@unsupervised_bp.route("/hierarchical", methods=["GET", "POST"])
@login_required
def hierarchical():
    df = load_current_dataframe()
    if df is None:
        flash("Aucun dataset chargé.", "warning")
        return redirect(url_for("data.upload"))

    info = get_dataset_info(df)
    numeric_cols = info.get("numeric_columns", [])
    result = None
    error = None

    if request.method == "POST":
        feature_cols = request.form.getlist("feature_cols")
        n_clusters = int(request.form.get("n_clusters", 3))
        linkage_method = request.form.get("linkage_method", "ward")

        if len(feature_cols) < 1:
            error = "Sélectionnez au moins 1 variable."
        else:
            try:
                result = hierarchical_clustering(df, feature_cols, n_clusters, linkage_method)
                _save_analysis(result, {"features": feature_cols, "n_clusters": n_clusters, "linkage": linkage_method})
                flash("CAH effectuée avec succès.", "success")
            except Exception as e:
                error = str(e)

    return render_template("analysis/unsupervised.html",
                           method="hierarchical", numeric_cols=numeric_cols,
                           result=result, error=error)
