from datetime import datetime
import json
from app import db


class Analysis(db.Model):
    __tablename__ = "analyses"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey("datasets.id"), nullable=False)
    analysis_type = db.Column(db.String(100), nullable=False)
    params = db.Column(db.Text)   # JSON
    results_summary = db.Column(db.Text)  # JSON résumé (sans les plots)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_params(self, params_dict):
        self.params = json.dumps(params_dict, ensure_ascii=False)

    def get_params(self):
        return json.loads(self.params) if self.params else {}

    def set_results(self, results_dict):
        # On ne stocke pas les plots (trop lourd) — juste les métriques
        summary = {k: v for k, v in results_dict.items() if k != "plot"}
        self.results_summary = json.dumps(summary, ensure_ascii=False)

    def get_results(self):
        return json.loads(self.results_summary) if self.results_summary else {}

    def __repr__(self):
        return f"<Analysis {self.analysis_type} - {self.created_at}>"
