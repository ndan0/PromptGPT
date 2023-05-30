
docker build -t test .  
docker run -d -p 80:8080 -e AIP_HTTP_PORT=8080 -e AIP_HEALTH_ROUTE=/health -e AIP_PREDICT_ROUTE=/predict test