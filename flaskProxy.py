from flask import Flask, request, jsonify
import logging
import requests

logging.basicConfig(filename='record.log', level=logging.INFO)
app = Flask(name)

URL = ""


@app.route("/", methods=['GET', 'POST'])
def chat():
    data = request.get_json()
    app.logger.info(data)
    r = requests.post(URL, json={"prompt": data["prompt"], "conversation": "", "options": {}})
    app.logger.info(r.json())
    return r.json()