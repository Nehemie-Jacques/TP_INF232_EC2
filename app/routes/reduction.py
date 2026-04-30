from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
from app import db
from app.models.analysis import Analysis
from app.services.data_service import load_current_dataframe, get_dataset_info, get_current_dataset_filename
from app.services.reduction_service import apply_pca, apply_tsne, apply_lda

reduction_bp = Blueprint("reduction", __name__, url_prefix="/analysis/reduction")


def _save_analysis(result, params):
    from app.models.dataset import Dataset
    filename = get_current_dataset_filename()
    ds = Dataset.query.filter_by(filename=filename, user_id=current_user.id).first()
    if ds:
        a = Analysis(user_id=current_user.id, dataset_id=ds.id,
                     analysis_type=f"reduction_{result['method'].lower()}")
        a.set_params(params)
        a.set_results(result)
        db.session.add(a)
        db.session.commit()


@reduction_bp.route("/pca", methods=["GET", "POST"])
@login_required
def pca():
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
        n_components = request.form.get("n_components", "")
        n_components = int(n_components) if n_components.isdigit() else None

        if len(feature_cols) < 2:
            error = "Sélectionnez au moins 2 variables."
        else:
            try:
                result = apply_pca(df, feature_cols, n_components)
                _save_analysis(result, {"features": feature_cols, "n_components": n_components})
                flash("ACP effectuée avec succès.", "success")
            except Exception as e:
                error = str(e)

    return render_template("analysis/reduction.html",
                           method="pca", numeric_cols=numeric_cols,
                           result=result, error=error)


@reduction_bp.route("/tsne", methods=["GET", "POST"])
@login_required
def tsne():
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
        perplexity = int(request.form.get("perplexity", 30))
        n_iter = int(request.form.get("n_iter", 1000))

        if len(feature_cols) < 2:
            error = "Sélectionnez au moins 2 variables."
        else:
            try:
                result = apply_tsne(df, feature_cols, perplexity=perplexity, n_iter=n_iter)
                _save_analysis(result, {"features": feature_cols, "perplexity": perplexity, "n_iter": n_iter})
                flash("t-SNE effectué avec succès.", "success")
            except Exception as e:
                error = str(e)

    return render_template("analysis/reduction.html",
                           method="tsne", numeric_cols=numeric_cols,
                           result=result, error=error)


@reduction_bp.route("/lda", methods=["GET", "POST"])
@login_required
def lda():
    df = load_current_dataframe()
    if df is None:
        flash("Aucun dataset chargé.", "warning")
        return redirect(url_for("data.upload"))

    info = get_dataset_info(df)
    numeric_cols = info.get("numeric_columns", [])
    all_cols = info.get("columns", [])
    result = None
    error = None

    if request.method == "POST":
        feature_cols = request.form.getlist("feature_cols")
        target_col = request.form.get("target_col")

        if len(feature_cols) < 1 or not target_col:
            error = "Sélectionnez des features et une variable cible."
        else:
            try:
                result = apply_lda(df, feature_cols, target_col)
                _save_analysis(result, {"features": feature_cols, "target": target_col})
                flash("LDA effectuée avec succès.", "success")
            except Exception as e:
                error = str(e)

    return render_template("analysis/reduction.html",
                           method="lda", numeric_cols=numeric_cols,
                           all_cols=all_cols, result=result, error=error)
