gcloud auth login

gcloud builds submit --config build.yaml

gcloud ai models upload --container-ports=80 --container-predict-route="/predict" --container-health-route="/health" --region=us-central1 --display-name=prompt_gpt	 --container-image-uri=us-central1-docker.pkg.dev/aerobic-gantry-387923/promptgpt/promptgpt_api