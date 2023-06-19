gcloud auth login

gcloud builds submit --config build.yaml

gcloud ai models upload --region=us-central1 --display-name=prompt_gpt_v2 --container-image-uri=us-central1-docker.pkg.dev/aerobic-gantry-387923/promptgpt/promptgpt_api --container-predict-route=/predict --container-health-route=/health --container-ports=8080




gcloud ai endpoints deploy-model prompt_gpt --display-name=prompt_gpt --region=us-central1 --model=5703556040390344704 --machine-type=n1-standard-4 --accelerator=type=nvidia-tesla-t4,count=1