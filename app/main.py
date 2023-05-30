import os


from fastapi import Request, FastAPI, Response
from fastapi.responses import JSONResponse

#####################
# Model Definition
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_folder = "models/pythia-70m"

from transformers import GPTNeoXForCausalLM, AutoTokenizer

model = GPTNeoXForCausalLM.from_pretrained(
  model_folder,
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

tokenizer = AutoTokenizer.from_pretrained(
  model_folder,
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

inputs = tokenizer("Hello, I am", return_tensors="pt")
tokens = model.generate(**inputs)
tokenizer.decode(tokens[0])

####################
# API Definition
app = FastAPI(title="PromptGPT")


@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {}


@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
  body = await request.json()

  prompt = body['instances'][0]['text']
  print(prompt)
  # prediction
  inputs = tokenizer(prompt, return_tensors="pt")
  tokens = model.generate(**inputs)
  answer = tokenizer.decode(tokens[0])
  # post-processing

  return {"predictions": [{"revised": "answer","confidence": 1.0}]}


if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0",port=8080)