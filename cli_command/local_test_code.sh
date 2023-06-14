
#Local build and test(Not recommended)
docker build -t test .  
docker run -d -p 80:8080 -e AIP_HTTP_PORT=8080 -e AIP_HEALTH_ROUTE=/health -e AIP_PREDICT_ROUTE=/predict test

##Build using GAR, pull it and do a local test(recommended)
gcloud builds submit --config build.yaml
docker pull us-central1-docker.pkg.dev/aerobic-gantry-387923/promptgpt/promptgpt_api
docker run -it -p 8080:8080 us-central1-docker.pkg.dev/aerobic-gantry-387923/promptgpt/promptgpt_api
#docker run -d -p 80:8080 -e AIP_HTTP_PORT=8080 -e AIP_HEALTH_ROUTE=/health -e AIP_PREDICT_ROUTE=/predict us-central1-docker.pkg.dev/aerobic-gantry-387923/promptgpt/promptgpt_api


############

curl -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json" https://us-central1-aiplatform.googleapis.com/v1/projects/9073376660593573888/locations/us-central1/endpoints/aerobic-gantry-387923:predict -d inferenceTest.json