import io
import json
import csv
from flask import Blueprint, request, send_file, flash, redirect, url_for, jsonify
from flask_login import login_required, current_user
from app.services.data_service import load_current_dataframe, get_descriptive_stats, get_dataset_info

export_bp = Blueprint("export", __name__, url_prefix="/export")


@export_bp.route("/csv")
@login_required
def export_csv():
    df = load_current_dataframe()
    if df is None:
        flash("Aucun dataset chargé.", "warning")
        return redirect(url_for("data.dashboard"))

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="dataset_export.csv"
    )


@export_bp.route("/stats/json")
@login_required
def export_stats_json():
    df = load_current_dataframe()
    if df is None:
        return jsonify({"error": "Aucun dataset"}), 400

    stats = get_descriptive_stats(df)
    info = get_dataset_info(df)

    export_data = {
        "info": {
            "n_rows": info["n_rows"],
            "n_cols": info["n_cols"],
            "columns": info["columns"],
            "missing_values": info["missing_values"],
        },
        "descriptive_statistics": stats
    }

    output = io.BytesIO(json.dumps(export_data, indent=2, ensure_ascii=False).encode("utf-8"))
    return send_file(
        output,
        mimetype="application/json",
        as_attachment=True,
        download_name="statistiques.json"
    )


@export_bp.route("/stats/pdf")
@login_required
def export_stats_pdf():
    df = load_current_dataframe()
    if df is None:
        flash("Aucun dataset chargé.", "warning")
        return redirect(url_for("data.dashboard"))

    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Table,
                                        TableStyle, Spacer, HRFlowable)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from datetime import datetime

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("Title", parent=styles["Title"],
                                     fontSize=18, spaceAfter=6,
                                     textColor=colors.HexColor("#185FA5"))
        subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"],
                                        fontSize=11, spaceAfter=12,
                                        textColor=colors.HexColor("#5F5E5A"))
        section_style = ParagraphStyle("Section", parent=styles["Heading2"],
                                       fontSize=13, spaceBefore=12, spaceAfter=6,
                                       textColor=colors.HexColor("#0F6E56"))

        info = get_dataset_info(df)
        stats = get_descriptive_stats(df)

        content = []

        # Titre
        content.append(Paragraph("Rapport d'Analyse Descriptive", title_style))
        content.append(Paragraph(
            f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} — INF232 EC2",
            subtitle_style
        ))
        content.append(HRFlowable(width="100%", thickness=1,
                                   color=colors.HexColor("#185FA5"), spaceAfter=12))

        # Informations dataset
        content.append(Paragraph("Informations du Dataset", section_style))
        info_data = [
            ["Propriété", "Valeur"],
            ["Nombre de lignes", str(info["n_rows"])],
            ["Nombre de colonnes", str(info["n_cols"])],
            ["Valeurs manquantes", str(info["missing_values"])],
            ["Colonnes numériques", str(len(info["numeric_columns"]))],
            ["Colonnes catégorielles", str(len(info["categorical_columns"]))],
        ]
        t = Table(info_data, colWidths=[7*cm, 9*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#185FA5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#f0f4fa"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        content.append(t)
        content.append(Spacer(1, 0.4*cm))

        # Statistiques descriptives
        if stats:
            content.append(Paragraph("Statistiques Descriptives", section_style))
            headers = ["Variable", "N", "Moyenne", "Écart-type", "Min", "Q1", "Médiane", "Q3", "Max"]
            stat_data = [headers]
            for col, s in stats.items():
                stat_data.append([
                    col[:20], str(s["count"]),
                    str(s["mean"]), str(s["std"]),
                    str(s["min"]), str(s["q25"]),
                    str(s["median"]), str(s["q75"]), str(s["max"])
                ])

            col_widths = [4.5*cm] + [1.7*cm] * 8
            t2 = Table(stat_data, colWidths=col_widths)
            t2.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F6E56")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#e8f5f0"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ]))
            content.append(t2)

        # Footer
        content.append(Spacer(1, 1*cm))
        content.append(HRFlowable(width="100%", thickness=0.5,
                                   color=colors.HexColor("#cccccc"), spaceAfter=6))
        content.append(Paragraph(
            "INF232 EC2 — Application de Collecte et Analyse de Données",
            ParagraphStyle("Footer", parent=styles["Normal"],
                           fontSize=8, textColor=colors.gray,
                           alignment=TA_CENTER)
        ))

        doc.build(content)
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="rapport_analyse.pdf"
        )

    except ImportError:
        flash("ReportLab non installé. Installez-le avec : pip install reportlab", "danger")
        return redirect(url_for("data.preview"))
    except Exception as e:
        flash(f"Erreur lors de la génération PDF : {str(e)}", "danger")
        return redirect(url_for("data.preview"))
