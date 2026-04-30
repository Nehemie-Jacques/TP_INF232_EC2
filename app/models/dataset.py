from datetime import datetime
from app import db


class Dataset(db.Model):
    __tablename__ = "datasets"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    filename = db.Column(db.String(300), nullable=False)
    original_name = db.Column(db.String(300), nullable=False)
    n_rows = db.Column(db.Integer, default=0)
    n_cols = db.Column(db.Integer, default=0)
    columns_info = db.Column(db.Text)  # JSON string
    source = db.Column(db.String(50), default="upload")  # upload | form | api
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    analyses = db.relationship("Analysis", backref="dataset", lazy="dynamic", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Dataset {self.name} ({self.n_rows}x{self.n_cols})>"
