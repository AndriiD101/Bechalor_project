from flask import Flask, send_from_directory
from flask_cors import CORS
import os

from database import create_tables
from routers.game import game_bp

app = Flask(__name__, static_folder=None)
CORS(app)

create_tables()

app.register_blueprint(game_bp)

# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path and os.path.exists(os.path.join(frontend_dir, path)):
        return send_from_directory(frontend_dir, path)
    return send_from_directory(frontend_dir, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
