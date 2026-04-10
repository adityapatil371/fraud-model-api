# Fraud Model API

Production API for the fraud detection model built in Level 3.

## Stack
- MLflow — experiment tracking
- FastAPI — model serving
- Docker — containerisation
- GitHub Actions — CI/CD

## How to Run

**Local development:**
```bash
pip install -r requirements.txt
make serve
```

**Train model:**
```bash
make train
```

**Docker:**
```bash
make docker-build
make docker-run
```

**Full stack (API + MLflow):**
```bash
make compose-up
```

## Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 0, "V1": -1.36, ..., "Amount": 149.62}'
```

## What I Learned
- MLflow tracks parameters, metrics, and artifacts from every training run so any model can be reproduced
- FastAPI serves models as REST APIs — Pydantic validates input before it reaches the model
- Docker packages code + environment into a container that runs identically anywhere
- Docker layer caching means copying requirements.txt before code saves rebuild time
- GitHub Actions runs automated tests on every push — CI catches broken code before it reaches production

## What Broke and How I Fixed It
- the VSCode save issue: there was an issue where cmd+s was not saving in vscode, to fix this i restrted the entire process properly
- the Docker registry issue: there was a docker issue where fastapi was not connecting with model fixed that
- the CI models/ folder issue: there was an issue where githuug required a folder to save model which only existed in system, fixed that be creating a folder if doesnt exist

## What I'd Do Differently
- redo the docker-compose for better results