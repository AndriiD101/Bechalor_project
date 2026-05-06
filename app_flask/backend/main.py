from flask import Flask, send_from_directory, request
from flask_cors import CORS
import os

from database import create_tables
from routers.game import game_bp

app = Flask(__name__, static_folder=None)
CORS(app)

create_tables()

# Register blueprint FIRST - this ensures API routes are registered before catch-all
app.register_blueprint(game_bp)

# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")

# Serve specific static files
@app.route("/")
@app.route("/index.html")
def serve_index():
    return send_from_directory(frontend_dir, "index.html")

@app.route("/main.js")
def serve_main_js():
    return send_from_directory(frontend_dir, "main.js")

@app.route("/styles.css")
def serve_styles():
    return send_from_directory(frontend_dir, "styles.css")

# Catch-all for SPA routing - serves index.html for unknown routes
@app.route("/<path:path>")
def serve_spa(path):
    # Don't serve frontend for API calls that weren't matched above
    if path.startswith("api/"):
        return {"error": "Not found"}, 404
    # Serve index.html for frontend SPA routing
    return send_from_directory(frontend_dir, "index.html")

@app.errorhandler(404)
def handle_404(e):
    # For API routes, return JSON 404
    if request.path.startswith("/api/"):
        return {"error": "Not found"}, 404
    # For frontend routes, serve index.html
    return send_from_directory(frontend_dir, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
