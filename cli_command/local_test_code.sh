
docker build -t test .  
docker run -d -p 80:8080 -e AIP_HTTP_PORT=8080 -e AIP_HEALTH_ROUTE=/health -e AIP_PREDICT_ROUTE=/predict test




############

curl -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json" https://us-central1-aiplatform.googleapis.com/v1/projects/9073376660593573888/locations/us-central1/endpoints/aerobic-gantry-387923:predict -d inferenceTest.json