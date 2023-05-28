#FastAPI Stuff
from typing import Union
from fastapi import FastAPI


#Dolly Stuff
import torch
from app.instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Init model...")

model_name = "databricks/dolly-v2-3b"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Please don't abuse this API."}

@app.post("/generate/")
def create_item(prompt: str):
    
    return generate_text(prompt=prompt)
