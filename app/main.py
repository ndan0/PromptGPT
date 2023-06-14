import os

import dolly

from flask import Flask, request, Response, jsonify
from flask_cors import CORS

#Init the model
model = dolly.Pipeline()

# API Definition
app = Flask(__name__)
CORS(app)


@app.route("/health")
def is_alive():
    print("/isalive request")
    status_code = Response(status=200)
    return status_code


@app.route("/predict", methods=["POST"])
def predict():
  print("/predict request")
  body = request.get_json()

  prompt = body['instances'][0]['text']
  print(prompt)
  answer = model.generate(prompt)
  # post-processing

  return {"predictions": [{"revised": answer,"confidence": 1.0}]}


if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0",port=8080)