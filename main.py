import os
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel

from fastapi import Request, FastAPI, Response
from fastapi.responses import JSONResponse

#Importing the AI model
import torch
from transformers import *

####################
# Model Definition
import torch
from transformers import pipeline

generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

class Prediction(BaseModel):
  revised: str 
  confidence: Optional[float]

class Predictions(BaseModel):
    predictions: List[Prediction]



####################
# API Definition
app = FastAPI(title="PromptGPT")

# #!Remove in production
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["POST"],
#     allow_headers=["Content-Type"],
#     allow_credentials=True,
# )
# #!Remove in production


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
  res = generate_text("Explain to me the difference between nuclear fission and fusion.")
  answer = res[0]["generated_text"]
  print(answer)

  # post-processing
  return Predictions(predictions=[{'revised': answer, 'confidence': 1.0}])

if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0",port=8080)