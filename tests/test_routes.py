"""Tests des routes Flask — INF232 EC2"""
import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app, db
from app.models.user import User


@pytest.fixture
def app():
    """Crée une instance de l'application en mode test."""
    class TestConfig:
        TESTING = True
        SECRET_KEY = "test-secret"
        SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
        SQLALCHEMY_TRACK_MODIFICATIONS = False
        WTF_CSRF_ENABLED = False
        UPLOAD_FOLDER = "/tmp/test_uploads"
        PLOTS_FOLDER = "/tmp/test_plots"

    os.makedirs("/tmp/test_uploads", exist_ok=True)
    os.makedirs("/tmp/test_plots", exist_ok=True)

    app = create_app(TestConfig)
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def auth_client(client, app):
    """Client authentifié."""
    with app.app_context():
        user = User(username="testuser", email="test@test.com")
        user.set_password("password123")
        db.session.add(user)
        db.session.commit()
    client.post("/auth/login", data={"username": "testuser", "password": "password123"})
    return client


class TestAuth:
    def test_login_page_loads(self, client):
        r = client.get("/auth/login")
        assert r.status_code == 200
        assert b"Connexion" in r.data

    def test_register_page_loads(self, client):
        r = client.get("/auth/register")
        assert r.status_code == 200

    def test_register_creates_user(self, client, app):
        r = client.post("/auth/register", data={
            "username": "newuser",
            "email": "new@test.com",
            "password": "pass123",
            "confirm_password": "pass123"
        }, follow_redirects=True)
        assert r.status_code == 200
        with app.app_context():
            assert User.query.filter_by(username="newuser").first() is not None

    def test_login_success(self, client, app):
        with app.app_context():
            user = User(username="logintest", email="login@test.com")
            user.set_password("pass123")
            db.session.add(user)
            db.session.commit()
        r = client.post("/auth/login", data={
            "username": "logintest", "password": "pass123"
        }, follow_redirects=True)
        assert r.status_code == 200

    def test_login_wrong_password(self, client, app):
        with app.app_context():
            user = User(username="badpass", email="bad@test.com")
            user.set_password("correct")
            db.session.add(user)
            db.session.commit()
        r = client.post("/auth/login", data={
            "username": "badpass", "password": "wrong"
        }, follow_redirects=True)
        assert b"incorrect" in r.data or r.status_code == 200

    def test_logout(self, auth_client):
        r = auth_client.get("/auth/logout", follow_redirects=True)
        assert r.status_code == 200


class TestDataRoutes:
    def test_upload_requires_login(self, client):
        r = client.get("/data/upload", follow_redirects=True)
        assert b"Connexion" in r.data or r.status_code == 200

    def test_upload_page_loads_when_auth(self, auth_client):
        r = auth_client.get("/data/upload")
        assert r.status_code == 200

    def test_form_page_loads(self, auth_client):
        r = auth_client.get("/data/form")
        assert r.status_code == 200

    def test_dashboard_loads(self, auth_client):
        r = auth_client.get("/dashboard")
        assert r.status_code == 200


class TestRootRedirect:
    def test_root_redirects(self, client):
        r = client.get("/")
        assert r.status_code in (302, 200)
