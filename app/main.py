import os


from fastapi import Request, FastAPI, Response
from fastapi.responses import JSONResponse

#####################
# Model Definition
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_folder = "test-pythia-70m"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import GPTNeoXForCausalLM, AutoTokenizer

model = GPTNeoXForCausalLM.from_pretrained(
  model_folder,
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
  device_map = "auto" # you are moving the device to the GPU
  
)

print(model.device)

tokenizer = AutoTokenizer.from_pretrained(
  model_folder,
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

inputs = tokenizer("Hello, I am", return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs.get("attention_mask", None)
tokens = model.generate(
  input_ids = input_ids.to(model.device),
  attention_mask=attention_mask.to(model.device) if attention_mask is not None else None,
)
tokenizer.decode(tokens[0] , skip_special_tokens=True)

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
  inpuit_ids = inputs["input_ids"]
  attention_mask = inputs.get("attention_mask", None)
  tokens = model.generate(
    input_ids = inpuit_ids.to(model.device),
    attention_mask=attention_mask.to(model.device) if attention_mask is not None else None,
  )
  answer = tokenizer.decode(tokens[0], skip_special_tokens=True)
  # post-processing

  return {"predictions": [{"revised": answer,"confidence": 1.0}]}


if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0",port=8080)