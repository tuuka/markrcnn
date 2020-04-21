from flask import Flask
from flask_cors import CORS
from app.model_ext import Model

application = Flask(__name__)
CORS(application)
torch_model = Model(application)

from app import routes