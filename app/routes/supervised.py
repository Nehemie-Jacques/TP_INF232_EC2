from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
from app import db
from app.models.analysis import Analysis
from app.services.data_service import load_current_dataframe, get_dataset_info, get_current_dataset_filename
from app.services.supervised_service import classify

supervised_bp = Blueprint("supervised", __name__, url_prefix="/analysis/supervised")


def _save_analysis(result, params):
    from app.models.dataset import Dataset
    filename = get_current_dataset_filename()
    ds = Dataset.query.filter_by(filename=filename, user_id=current_user.id).first()
    if ds:
        a = Analysis(user_id=current_user.id, dataset_id=ds.id,
                     analysis_type=f"supervised_{result['model_name']}")
        a.set_params(params)
        a.set_results(result)
        db.session.add(a)
        db.session.commit()


@supervised_bp.route("/<model_name>", methods=["GET", "POST"])
@login_required
def run_model(model_name):
    VALID_MODELS = ["knn", "svm", "random_forest", "decision_tree"]
    if model_name not in VALID_MODELS:
        flash("Modèle inconnu.", "danger")
        return redirect(url_for("data.dashboard"))

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
        test_size = float(request.form.get("test_size", 0.2))

        # Paramètres spécifiques au modèle
        params = {"test_size": test_size}
        if model_name == "knn":
            params["k"] = request.form.get("k", 5)
            params["weights"] = request.form.get("weights", "uniform")
            params["metric"] = request.form.get("metric", "minkowski")
        elif model_name == "svm":
            params["kernel"] = request.form.get("kernel", "rbf")
            params["C"] = request.form.get("C", 1.0)
            params["gamma"] = request.form.get("gamma", "scale")
        elif model_name == "random_forest":
            params["n_estimators"] = request.form.get("n_estimators", 100)
            params["max_depth"] = request.form.get("max_depth", None)
        elif model_name == "decision_tree":
            params["max_depth"] = request.form.get("max_depth", 5)
            params["criterion"] = request.form.get("criterion", "gini")

        if not feature_cols or not target_col:
            error = "Sélectionnez au moins 1 feature et 1 variable cible."
        elif target_col in feature_cols:
            error = "La variable cible ne peut pas être un prédicteur."
        else:
            try:
                result = classify(df, feature_cols, target_col, model_name, test_size, params)
                _save_analysis(result, {**params, "features": feature_cols, "target": target_col})
                flash(f"Classification {model_name.replace('_',' ').title()} effectuée.", "success")
            except Exception as e:
                error = str(e)

    model_labels = {
        "knn": "K-Nearest Neighbors",
        "svm": "Support Vector Machine",
        "random_forest": "Random Forest",
        "decision_tree": "Arbre de Décision",
    }

    return render_template("analysis/supervised.html",
                           model_name=model_name,
                           model_label=model_labels[model_name],
                           numeric_cols=numeric_cols,
                           all_cols=all_cols,
                           result=result,
                           error=error)
