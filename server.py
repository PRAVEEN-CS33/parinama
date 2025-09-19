from flask import Flask
from flask_cors import CORS
from app.config import Config

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Import routes from routes.py
from app.routes import *

if __name__ == "__main__":
    app.run(debug=True)