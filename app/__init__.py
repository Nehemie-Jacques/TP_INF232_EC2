import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect
from config import Config

db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
csrf = CSRFProtect()


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    csrf.init_app(app)

    login_manager.login_view = "auth.login"
    login_manager.login_message = "Connectez-vous pour accéder à cette page."
    login_manager.login_message_category = "warning"

    # Créer les dossiers nécessaires
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["PLOTS_FOLDER"], exist_ok=True)

    # Blueprints
    from app.routes.auth import auth_bp
    from app.routes.data import data_bp
    from app.routes.regression import regression_bp
    from app.routes.reduction import reduction_bp
    from app.routes.supervised import supervised_bp
    from app.routes.unsupervised import unsupervised_bp
    from app.routes.export import export_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(regression_bp)
    app.register_blueprint(reduction_bp)
    app.register_blueprint(supervised_bp)
    app.register_blueprint(unsupervised_bp)
    app.register_blueprint(export_bp)

    # Route principale
    from flask import redirect, url_for
    from flask_login import current_user

    @app.route("/")
    def index():
        if current_user.is_authenticated:
            return redirect(url_for("data.dashboard"))
        return redirect(url_for("auth.login"))

    # Health check endpoint for external load balancers / Render
    @app.route('/ping')
    def ping():
        return 'pong', 200

    # Gestion des erreurs
    @app.errorhandler(404)
    def not_found(e):
        from flask import render_template
        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def server_error(e):
        from flask import render_template
        return render_template("errors/500.html"), 500

    return app
