# PromptGPT
- You can access our [website front end repo](https://github.com/arihanv/PromptGPT) or our [extension](https://github.com/arihanv/PromptGPT-Ext)

- If you are interested in training the model. You can access it in the [training folder](https://github.com/DanNguyenN/PromptGPT/tree/main/training)


- You can build the container and run it locally assume you have GPU. This is for the backend only. 
    - You have to download dolly-v2-3b model locally and put it in app/models folder
        - You can also change ```PRETRAINED_MODEL_NAME_OR_PATH ``` in [app/main.py](https://github.com/DanNguyenN/PromptGPT/blob/main/app/main.py)
    - You can test it by editing the inferenceTest.json and then do curl command using that json file to 0.0.0.0:8080/predict
- If you want to deploy on the cloud, 
    - upload the dolly-v2-3b model(https://huggingface.co/databricks/dolly-v2-3b) to Google Cloud Storage
    - fix the first step in build.yaml to be your GCS bucket
    - Then follow this document(https://cloud.google.com/vertex-ai/docs/general/deployment) to deploy to Vertex AI 
