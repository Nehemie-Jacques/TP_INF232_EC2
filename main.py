from app import create_app
from app.models.user import User
from app.models.dataset import Dataset
from app.models.analysis import Analysis

app = create_app()


@app.shell_context_processor
def make_shell_context():
    return {"db": db, "User": User, "Dataset": Dataset, "Analysis": Analysis}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
