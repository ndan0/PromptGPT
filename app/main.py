import os

from flask import Flask, request, Response, jsonify
from flask_cors import CORS

import torch.multiprocessing as mp

# Set the start method to 'spawn'
mp.set_start_method('spawn')









PRETRAINED_MODEL_NAME_OR_PATH = "models/dolly_base"
#PRETRAINED_MODEL_NAME_OR_PATH = "databricks/dolly-v2-3b"
repo_name = "Rami/dolly_prompt_generator"

#Init the model

from rich import print
import logging
from pathlib import Path
logger = logging.getLogger(__name__)
#ROOT_PATH = Path(__file__).parent.parent
import session_info
session_info.show()


# # Check the GPU env
# 1. You can check the GPU in the Google Colab by clicking  and efficieny
# 2. Check if the GPU can use bfloat16 most effective as most model are pre-trained with bfloat16

import torch
from rich import print

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Cuda capability: ", torch.cuda.get_device_capability(0))
    '''
    On pre-ampere hardware bf16 works, but doesn't provide speed-ups compared to fp32 matmul operations, and some matmul operations are failing outright, so this check is more like "guaranteed to work and be performant" than "works somehow".  https://github.com/pytorch/pytorch/issues/75427
    '''
    print(f"bfloat16 support: { torch.cuda.is_bf16_supported()}") 


# # Set the Seed Environment of the Notebook to ensure the reproducibility


from transformers import set_seed

DEFAULT_SEED = 42

set_seed( DEFAULT_SEED )


# # Download the Tokenizers
# 1. We are suing Dolly model which was trained on the Pythia model. Instead we are recreating the dollvy tokenizer from the Pythia tokenizer


from transformers import AutoTokenizer

# Special Tokens
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"
DEFAULT_SEED = 42



eleutherai_python_3b = "EleutherAI/pythia-2.8b"
eleutherai_python_7b = "EleutherAI/pythia-6.9b"
dolly_v2_tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b")
print(dolly_v2_tokenizer)


# # Download the Model
# 1. Torch Datat

# ## Setup Bits and Butes Config

# ## 4 Bit Configuration


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "EleutherAI/gpt-neox-20b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit = False,
    load_in_8bit  = True,
    llm_int8_threshold = 6.0,
)


# ## Download the LM Models
# Then we have to apply some preprocessing to the model to prepare it for training. For that use the `prepare_model_for_kbit_training` method from PEFT.


from transformers import AutoModelForCausalLM
#assert torch.cuda.is_available(), "You need to have a GPU to run this notebook."
print(torch.cuda.is_available())
n_gpus = torch.cuda.device_count()
print(f"Number of GPUs: {n_gpus}")
def model_init():
    # free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    # max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'

    # n_gpus = torch.cuda.device_count()
    # print(f"Number of GPUs: {n_gpus}")
    # max_memory = {i: max_memory for i in range(n_gpus)}
    # print(f"Max memory: {max_memory}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path = PRETRAINED_MODEL_NAME_OR_PATH,
            trust_remote_code = True,
            use_cache = False,
            torch_dtype =  torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map = "auto",
            load_in_8bit = False,
            load_in_4bit = True,
            low_cpu_mem_usage = True, # low cpu memory usage is to be true when the device map is auto
            #max_memory =  max_memory,
            quantization_config = bnb_config,
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path = "databricks/dolly-v2-3b",
            trust_remote_code = True,
            use_cache = False,
            torch_dtype =  torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map = "auto",
            load_in_8bit = False,
            load_in_4bit = True,
            low_cpu_mem_usage = True, # low cpu memory usage is to be true when the device map is auto
            #max_memory =  max_memory,
            quantization_config = bnb_config,
        )
    return model

model = model_init()

print(model)


# # Text Generation inference

# # Text Generation COnfiguration


from transformers import AutoModelForCausalLM, GenerationConfig
import random
generation_config =  GenerationConfig(
    max_new_tokens = 256, # The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
    num_beams = 1, # 1 means no beam search instead greedy search
    temperature = random.uniform(0.01 , .98), # Parameters for manipulation of the model output logits
    top_p = 0.92, # Parameters for manipulation of the model output logits
    top_k = 50, # Parameters to only select the top-k tokens, instead of sampling from the distribution
    do_sample = True ,# select a random token from the top-k tokens (set to 0 to disable top-k sampling) instead of choosing the one with the highest probability
    use_cache = True, # Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    repetition_penalty = 1.02, # The parameter for repetition penalty. 1.0 means no penalty. See this paper for more details.
)


# ## Download the adaper config if not present


from peft import PeftConfig, PeftModel


config = PeftConfig.from_pretrained(repo_name) 


# ## Combine the Model and Adapter


from peft import PeftConfig, PeftModel

inference_model = None
try:
    inference_model = PeftModel.from_pretrained(
        model,
        repo_name,
    )
except NameError as e:
    ## Donwload the model from the HFhub
    model = model_init()
    
    inference_model = PeftModel.from_pretrained(
        model,
        repo_name,
    )


# ## Create the Instruction Generation Pipeline


import logging
import re
from typing import List

import numpy as np
from transformers import Pipeline, PreTrainedTokenizer


logger = logging.getLogger(__name__)

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)

# This is the prompt that is used for generating responses using an already trained model.  It ends with the response
# key, where the job of the model is to provide the completion that follows it (i.e. the response itself).
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)


def get_special_token_id(tokenizer: PreTrainedTokenizer, key: str) -> int:
    """Gets the token ID for a given string that has been added to the tokenizer as a special token.
    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.
    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer
        key (str): the key to convert to a single token
    Raises:
        RuntimeError: if more than one ID was generated
    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        raise ValueError(f"Expected only a single token for '{key}' but found {token_ids}")
    return token_ids[0]

from transformers import AutoModelForCausalLM, GenerationConfig
class InstructionTextGenerationPipeline(Pipeline):
    def __init__(
        self, 
        generation_config: GenerationConfig = None,
        **kwargs,
    ):
        """Initialize the pipeline
        Args:
            do_sample (bool, optional): Whether or not to use sampling. Defaults to True.
            max_new_tokens (int, optional): Max new tokens after the prompt to generate. Defaults to 128.
            top_p (float, optional): If set to float < 1, only the smallest set of most probable tokens with
                probabilities that add up to top_p or higher are kept for generation. Defaults to 0.92.
            top_k (int, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering.
                Defaults to 0.
        """
        self.generation_config: GenerationConfig = generation_config
        super().__init__(**kwargs)

    def _sanitize_parameters(self,
                            return_full_text: bool = None,
                            **generate_kwargs):
        preprocess_params = {}
        assert self.generation_config is not None, "Generation config is not initialized."

        # newer versions of the tokenizer configure the response key as a special token.  newer versions still may
        # append a newline to yield a single token.  find whatever token is configured for the response key.
        tokenizer_response_key = next(
            (token for token in self.tokenizer.additional_special_tokens if token.startswith(RESPONSE_KEY)), None
        )

        response_key_token_id = None
        end_key_token_id = None
        if tokenizer_response_key:
            try:
                response_key_token_id = get_special_token_id(self.tokenizer, tokenizer_response_key)
                end_key_token_id = get_special_token_id(self.tokenizer, END_KEY)

                # Ensure generation stops once it generates "### End"
                generate_kwargs["eos_token_id"] = end_key_token_id
                self.generation_config.eos_token_id = end_key_token_id
            except ValueError:
                pass

        forward_params = generate_kwargs
        postprocess_params = {
            "response_key_token_id": response_key_token_id,
            "end_key_token_id": end_key_token_id
        }

        if return_full_text is not None:
            postprocess_params["return_full_text"] = return_full_text
            print(postprocess_params)

        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, instruction_text, **generate_kwargs):
        prompt_text = PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction_text)
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
        )
        inputs["prompt_text"] = prompt_text
        inputs["instruction_text"] = instruction_text
        return inputs
    ## Only Once
    def _forward(self, model_inputs , eos_token_id):
        assert self.model is not None, "Model is not initialized."
        assert self.generation_config is not None, "Generation config is not initialized."
        assert self.tokenizer is not None, "Tokenizer is not initialized."
        assert self.tokenizer.pad_token_id is not None, "Tokenizer does not have a pad token ID."
        #print(self.generation_config)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)

        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]

        generated_sequence = self.model.generate(
            input_ids=input_ids.to(self.model.device),
            attention_mask=attention_mask.to(self.model.device) if attention_mask is not None else None,
            pad_token_id=self.tokenizer.pad_token_id,
            generation_config=self.generation_config,
        )

        out_b = generated_sequence.shape[0]
        if self.framework == "pt":
            generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])

        instruction_text = model_inputs.pop("instruction_text")
        return {"generated_sequence": generated_sequence, "input_ids": input_ids, "instruction_text": instruction_text}

    def postprocess(self, model_outputs, response_key_token_id, end_key_token_id, return_full_text: bool = False):

        generated_sequence = model_outputs["generated_sequence"][0]
        instruction_text = model_outputs["instruction_text"]

        generated_sequence: List[List[int]] = generated_sequence.numpy().tolist()
        records = []
        for sequence in generated_sequence:

            # The response will be set to this variable if we can identify it.
            decoded = None

            # If we have token IDs for the response and end, then we can find the tokens and only decode between them.
            if response_key_token_id and end_key_token_id:
                # Find where "### Response:" is first found in the generated tokens.  Considering this is part of the
                # prompt, we should definitely find it.  We will return the tokens found after this token.
                try:
                    response_pos = sequence.index(response_key_token_id)
                except ValueError:
                    logger.warn(f"Could not find response key {response_key_token_id} in: {sequence}")
                    response_pos = None

                if response_pos:
                    # Next find where "### End" is located.  The model has been trained to end its responses with this
                    # sequence (or actually, the token ID it maps to, since it is a special token).  We may not find
                    # this token, as the response could be truncated.  If we don't find it then just return everything
                    # to the end.  Note that even though we set eos_token_id, we still see the this token at the end.
                    try:
                        end_pos = sequence.index(end_key_token_id)
                    except ValueError:
                        end_pos = None

                    decoded = self.tokenizer.decode(sequence[response_pos + 1 : end_pos]).strip()

            if not decoded:
                # Otherwise we'll decode everything and use a regex to find the response and end.

                fully_decoded = self.tokenizer.decode(sequence)

                # The response appears after "### Response:".  The model has been trained to append "### End" at the
                # end.
                m = re.search(r"#+\s*Response:\s*(.+?)#+\s*End", fully_decoded, flags=re.DOTALL)

                if m:
                    decoded = m.group(1).strip()
                else:
                    # The model might not generate the "### End" sequence before reaching the max tokens.  In this case,
                    # return everything after "### Response:".
                    m = re.search(r"#+\s*Response:\s*(.+)", fully_decoded, flags=re.DOTALL)
                    if m:
                        decoded = m.group(1).strip()
                    else:
                        logger.warn(f"Failed to find response in:\n{fully_decoded}")

            # If the full text is requested, then append the decoded text to the original instruction.
            # This technically isn't the full text, as we format the instruction in the prompt the model has been
            # trained on, but to the client it will appear to be the full text.
            if return_full_text:
                decoded = f"{instruction_text}\n{decoded}"

            rec = {"generated_text": decoded}

            records.append(rec)

        return records


from rich import print
generate_text = InstructionTextGenerationPipeline(model=model,
                                                tokenizer = dolly_v2_tokenizer, 
                                                task="text-generation" , 
                                                return_full_text=True,
                                                generation_config=generation_config)
                                                
print(generate_text.task)
print(generate_text._sanitize_parameters())
#print(generate_text("### Instruction: What is the capital of France? ### Response:"))



# ## Langchain Prompt with Hugging Face Pipeline



from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

# template for an instrution with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

# template for an instruction with input
prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")

hf_pipeline = HuggingFacePipeline(pipeline=generate_text)

llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)

GENERATE_PROMPT_INSTRUCTION = "Given a prompt from the user that was meant to be feed into a GPT style model , rewrite the prompt that will improve the quality of the generated text."
        


# API Definition
app = Flask(__name__)
CORS(app)


@app.route("/health")
def is_alive():
    status_code = Response(status=200)
    return status_code


@app.route("/predict", methods=["POST"])
def predict():
  body = request.get_json()

  user_prompt = body['instances'][0]['text']
  
  answer = llm_context_chain.predict(instruction = GENERATE_PROMPT_INSTRUCTION,context=user_prompt).lstrip()
  
  print(user_prompt)
  print(answer)
  # post-processing

  return {"predictions": [{"revised": answer}]}


if __name__ == "__main__":
  app.run(debug=False, host="0.0.0.0",port=8080, threaded=False)