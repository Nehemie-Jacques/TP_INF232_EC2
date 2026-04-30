from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
from app import db
from app.models.analysis import Analysis
from app.services.data_service import load_current_dataframe, get_dataset_info
from app.services.regression_service import simple_regression, multiple_regression

regression_bp = Blueprint("regression", __name__, url_prefix="/analysis/regression")


def _save_analysis(result, dataset_filename, params):
    from app.models.dataset import Dataset
    ds = Dataset.query.filter_by(filename=dataset_filename, user_id=current_user.id).first()
    if ds:
        analysis = Analysis(
            user_id=current_user.id,
            dataset_id=ds.id,
            analysis_type=f"regression_{result['type']}",
        )
        analysis.set_params(params)
        analysis.set_results(result)
        db.session.add(analysis)
        db.session.commit()


@regression_bp.route("/simple", methods=["GET", "POST"])
@login_required
def simple():
    df = load_current_dataframe()
    if df is None:
        flash("Aucun dataset chargé.", "warning")
        return redirect(url_for("data.upload"))

    info = get_dataset_info(df)
    numeric_cols = info.get("numeric_columns", [])
    result = None
    error = None

    if request.method == "POST":
        x_col = request.form.get("x_col")
        y_col = request.form.get("y_col")
        test_size = float(request.form.get("test_size", 0.2))

        if not x_col or not y_col:
            error = "Sélectionnez les deux variables."
        elif x_col == y_col:
            error = "Les deux variables doivent être différentes."
        else:
            try:
                result = simple_regression(df, x_col, y_col, test_size)
                from app.services.data_service import get_current_dataset_filename
                _save_analysis(result, get_current_dataset_filename(),
                               {"x_col": x_col, "y_col": y_col, "test_size": test_size})
                flash("Régression simple effectuée avec succès.", "success")
            except Exception as e:
                error = str(e)

    return render_template("analysis/regression.html",
                           analysis_type="simple",
                           numeric_cols=numeric_cols,
                           result=result,
                           error=error)


@regression_bp.route("/multiple", methods=["GET", "POST"])
@login_required
def multiple():
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
        target_col = request.form.get("target_col")
        test_size = float(request.form.get("test_size", 0.2))

        if not feature_cols or not target_col:
            error = "Sélectionnez au moins 2 prédicteurs et 1 variable cible."
        elif target_col in feature_cols:
            error = "La variable cible ne peut pas être un prédicteur."
        else:
            try:
                result = multiple_regression(df, feature_cols, target_col, test_size)
                from app.services.data_service import get_current_dataset_filename
                _save_analysis(result, get_current_dataset_filename(),
                               {"features": feature_cols, "target": target_col, "test_size": test_size})
                flash("Régression multiple effectuée avec succès.", "success")
            except Exception as e:
                error = str(e)

    return render_template("analysis/regression.html",
                           analysis_type="multiple",
                           numeric_cols=numeric_cols,
                           result=result,
                           error=error)
