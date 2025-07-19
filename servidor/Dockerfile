# cpu-branch: Dockerfile
FROM python:3.10-slim

# 1) Dependencias de sistema para PIL / OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender1 \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Requirements CPU-only
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt

# 3) Tu código
COPY ia_detect.py .
COPY main.py .

EXPOSE 8000
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug", "--workers", "1"]

