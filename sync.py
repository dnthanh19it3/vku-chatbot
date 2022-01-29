import json

from flask import Flask, request
from flask_cors import CORS
import time

import train

app = Flask(__name__)
CORS(app)
app.config["DEBUG"] = True

@app.route("/sync")
def predict():
    try:
        train.train()
        return json.dumps({"status": "1", "message": "Train thành công!"})
    except:
        return json.dumps({"status": "0", "message": "Train thất bại!"})

app.run()