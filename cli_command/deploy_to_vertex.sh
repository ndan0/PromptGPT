gcloud auth login

gcloud builds submit --config build.yaml

gcloud ai models upload --region=us-central1 --display-name=prompt_gpt --container-image-uri=us-central1-docker.pkg.dev/aerobic-gantry-387923/promptgpt/promptgpt_api --container-predict-route=/predict --container-health-route=/health --container-ports=8080

gcloud ai endpoints deploy-model prompt_gpt --region=us-central1 --model=prompt_gpt --machine-type=n1-standard-4 --accelerator=count=1,type=nvidia-tesla-t4