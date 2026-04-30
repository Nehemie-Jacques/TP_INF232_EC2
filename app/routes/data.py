import json
import os
from flask import (Blueprint, render_template, redirect, url_for,
                   flash, request, session, current_app, jsonify)
from flask_login import login_required, current_user
from app import db
from app.models.dataset import Dataset
from app.services.data_service import (
    allowed_file, save_uploaded_file, load_dataframe,
    get_dataset_info, get_dataframe_preview, save_manual_data,
    set_current_dataset, load_current_dataframe, get_descriptive_stats
)
from app.services.plot_service import (
    plot_histogram, plot_correlation_matrix,
    plot_scatter, plot_distribution_grid
)
from app.models.analysis import Analysis

data_bp = Blueprint("data", __name__)


@data_bp.route("/dashboard")
@login_required
def dashboard():
    datasets = Dataset.query.filter_by(user_id=current_user.id)\
                            .order_by(Dataset.created_at.desc()).all()
    analyses = Analysis.query.filter_by(user_id=current_user.id)\
                             .order_by(Analysis.created_at.desc()).limit(5).all()
    current_ds = session.get("current_dataset")
    current_ds_obj = None
    if current_ds:
        current_ds_obj = Dataset.query.filter_by(
            user_id=current_user.id, filename=current_ds
        ).first()
    return render_template("dashboard.html",
                           datasets=datasets,
                           recent_analyses=analyses,
                           current_dataset=current_ds_obj)


@data_bp.route("/data/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            flash("Aucun fichier sélectionné.", "danger")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("Aucun fichier sélectionné.", "danger")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Format non supporté. Utilisez CSV, XLSX ou XLS.", "danger")
            return redirect(request.url)

        try:
            unique_name, original_name = save_uploaded_file(file)
            df = load_dataframe(unique_name)
            if df is None or df.empty:
                flash("Impossible de lire le fichier. Vérifiez son format.", "danger")
                return redirect(request.url)

            info = get_dataset_info(df)
            dataset_name = request.form.get("dataset_name", "").strip() or original_name

            dataset = Dataset(
                user_id=current_user.id,
                name=dataset_name,
                filename=unique_name,
                original_name=original_name,
                n_rows=info["n_rows"],
                n_cols=info["n_cols"],
                columns_info=json.dumps(info["columns"]),
                source="upload"
            )
            db.session.add(dataset)
            db.session.commit()

            set_current_dataset(unique_name)
            flash(f"Dataset « {dataset_name} » chargé avec succès "
                  f"({info['n_rows']} lignes, {info['n_cols']} colonnes).", "success")
            return redirect(url_for("data.preview"))

        except Exception as e:
            flash(f"Erreur lors du chargement : {str(e)}", "danger")

    return render_template("data/upload.html")


@data_bp.route("/data/form", methods=["GET", "POST"])
@login_required
def form():
    if request.method == "POST":
        try:
            columns_raw = request.form.get("columns", "").strip()
            rows_raw = request.form.get("rows_data", "").strip()

            if not columns_raw or not rows_raw:
                flash("Veuillez renseigner les colonnes et les données.", "danger")
                return redirect(request.url)

            columns = [c.strip() for c in columns_raw.split(",") if c.strip()]
            rows = []
            for line in rows_raw.strip().split("\n"):
                if line.strip():
                    vals = [v.strip() for v in line.split(",")]
                    while len(vals) < len(columns):
                        vals.append("")
                    rows.append(vals[:len(columns)])

            if not rows:
                flash("Aucune donnée valide saisie.", "danger")
                return redirect(request.url)

            unique_name, orig_name, df = save_manual_data(rows, columns)
            info = get_dataset_info(df)
            dataset_name = request.form.get("dataset_name", "Saisie manuelle").strip()

            dataset = Dataset(
                user_id=current_user.id,
                name=dataset_name,
                filename=unique_name,
                original_name=orig_name,
                n_rows=info["n_rows"],
                n_cols=info["n_cols"],
                columns_info=json.dumps(info["columns"]),
                source="form"
            )
            db.session.add(dataset)
            db.session.commit()

            set_current_dataset(unique_name)
            flash(f"Données enregistrées ({info['n_rows']} lignes).", "success")
            return redirect(url_for("data.preview"))

        except Exception as e:
            flash(f"Erreur : {str(e)}", "danger")

    return render_template("data/form.html")


@data_bp.route("/data/preview")
@login_required
def preview():
    df = load_current_dataframe()
    if df is None:
        flash("Aucun dataset chargé. Veuillez d'abord importer des données.", "warning")
        return redirect(url_for("data.upload"))

    info = get_dataset_info(df)
    columns, rows = get_dataframe_preview(df, n=15)
    stats = get_descriptive_stats(df)

    # Graphiques
    dist_plot = plot_distribution_grid(df)
    corr_plot = plot_correlation_matrix(df)

    return render_template("data/preview.html",
                           info=info,
                           columns=columns,
                           rows=rows,
                           stats=stats,
                           dist_plot=dist_plot,
                           corr_plot=corr_plot)


@data_bp.route("/data/select/<int:dataset_id>")
@login_required
def select_dataset(dataset_id):
    dataset = Dataset.query.filter_by(
        id=dataset_id, user_id=current_user.id
    ).first_or_404()
    set_current_dataset(dataset.filename)
    flash(f"Dataset « {dataset.name} » sélectionné.", "success")
    return redirect(url_for("data.preview"))


@data_bp.route("/data/delete/<int:dataset_id>", methods=["POST"])
@login_required
def delete_dataset(dataset_id):
    dataset = Dataset.query.filter_by(
        id=dataset_id, user_id=current_user.id
    ).first_or_404()

    # Supprimer le fichier
    filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], dataset.filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    # Si c'est le dataset courant, on le désélectionne
    if session.get("current_dataset") == dataset.filename:
        session.pop("current_dataset", None)

    db.session.delete(dataset)
    db.session.commit()
    flash(f"Dataset « {dataset.name} » supprimé.", "info")
    return redirect(url_for("data.dashboard"))


@data_bp.route("/data/plot/histogram")
@login_required
def histogram_plot():
    df = load_current_dataframe()
    col = request.args.get("col")
    if df is None or not col or col not in df.columns:
        return jsonify({"error": "Paramètres invalides"}), 400
    try:
        plot = plot_histogram(df, col)
        return jsonify({"plot": plot})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@data_bp.route("/data/plot/scatter")
@login_required
def scatter_plot():
    df = load_current_dataframe()
    x_col = request.args.get("x")
    y_col = request.args.get("y")
    if df is None or not x_col or not y_col:
        return jsonify({"error": "Paramètres invalides"}), 400
    try:
        plot = plot_scatter(df, x_col, y_col)
        return jsonify({"plot": plot})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
