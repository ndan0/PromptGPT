FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-13.py310


# Install libraries
COPY ./requirements.txt ./

#Uninstall torchvision and torchaudio
RUN pip uninstall -y torchvision torchaudio

RUN pip install --no-cache-dir -r requirements.txt

# Setup container directories
RUN mkdir /app

# Copy local code to the container
COPY ./app /app

# launch server with gunicorn
WORKDIR /app
EXPOSE 8080
CMD ["gunicorn", "main:app", "--timeout=0", "--preload", \
     "--workers=1", "--threads=4", "--bind=0.0.0.0:8080"]