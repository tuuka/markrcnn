from flask import Flask
from flask_cors import CORS
from flask_caching import Cache


cache = Cache(config={'CACHE_TYPE': 'simple'})
application = Flask(__name__)
#cache.init_app(application)
CORS(application)

from app import routes