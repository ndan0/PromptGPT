import os
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel

from fastapi import Request, FastAPI, Response
from fastapi.responses import JSONResponse

####################
# Model Definition
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "test-dolly-v2-3b"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
#model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)

end_key_token_id = tokenizer.encode("### End")[0]

instruct_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer,pad_token_id=tokenizer.pad_token_id, eos_token_id=end_key_token_id)

def generate(instruction): 
    input_ids = tokenizer.encode(instruction, return_tensors="pt")
    input_ids = input_ids.to(model.device)  # Move input_ids to the same device as the model
    generated_output = model.generate(input_ids, max_length=256,pad_token_id=tokenizer.pad_token_id, eos_token_id=end_key_token_id)
    dd = tokenizer.decode(generated_output[0])
    return dd

print(generate("### Start\n Hello \n### End"))

class Prediction(BaseModel):
  revised: str 
  confidence: Optional[float]

class Predictions(BaseModel):
    predictions: List[Prediction]



####################
# API Definition
app = FastAPI(title="PromptGPT")

#!Remove in production
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
    allow_credentials=True,
)
#!Remove in production


AIP_HEALTH_ROUTE = os.environ.get('AIP_HEALTH_ROUTE', '/health')
AIP_PREDICT_ROUTE = os.environ.get('AIP_PREDICT_ROUTE', '/predict')

@app.get(AIP_HEALTH_ROUTE, status_code=200)
async def health():
    return {'health': 'ok'}
  
@app.post(AIP_PREDICT_ROUTE, 
          response_model=Predictions,
          response_model_exclude_unset=True)
async def predict(request: Request):
  # logic to load the model
  # pre-processing
  body = await request.json()
  prompt = body['instances'][0]['text']
  print(prompt)
  # prediction
  res = generate_text(prompt)
  answer = res[0]["generated_text"]
  print(answer)

  # post-processing
  return Predictions(predictions=[{'revised': answer, 'confidence': 1.0}])

if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0",port=8080)