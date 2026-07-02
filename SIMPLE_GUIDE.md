# Simple Module-Based Architecture Guide

## Overview

This project uses a **NestJS-inspired module structure** where each service is a single Python file that exports a FastAPI router. This keeps things simple and maintainable.

## Structure

```
.
├── main.py                    # Main app - registers all services
├── services/                  # Service modules
│   ├── deskew_service.py     # Deskew service
│   └── [add more here]       # Future services
└── tests/                     # Tests
    ├── test_main.py
    └── test_deskew.py
```

## How It Works

### 1. Each Service is a Module

Each service is a single `.py` file in the `services/` directory:

```python
# services/my_service.py
from fastapi import APIRouter

router = APIRouter(prefix="/my-service", tags=["my-service"])

@router.get("/")
async def my_endpoint():
    return {"message": "Hello"}

@router.get("/health")
async def health():
    return {"status": "healthy"}
```

### 2. Main App Registers Services

The `main.py` file imports and registers all service routers:

```python
# main.py
from fastapi import FastAPI
from services import my_service

app = FastAPI()

# Register service routers
app.include_router(my_service.router, prefix="/api/v1")
```

### 3. URL Structure

Services are automatically namespaced:

- Main app: `http://localhost:8000/`
- Service: `http://localhost:8000/api/v1/my-service/`
- Endpoint: `http://localhost:8000/api/v1/my-service/endpoint`

## Adding a New Service

### Step 1: Create Service File

Create `services/new_service.py`:

```python
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/new-service", tags=["new-service"])

class MyRequest(BaseModel):
    data: str

class MyResponse(BaseModel):
    result: str

@router.post("/", response_model=MyResponse)
async def process(request: MyRequest):
    # Your logic here
    return MyResponse(result=f"Processed: {request.data}")

@router.get("/health")
async def health():
    return {"service": "new-service", "status": "healthy"}
```

### Step 2: Register in Main App

Update `main.py`:

```python
from services import deskew_service, new_service  # Add import

app.include_router(deskew_service.router, prefix="/api/v1")
app.include_router(new_service.router, prefix="/api/v1")  # Add this line
```

### Step 3: Test It

```bash
# Start server
python main.py

# Test endpoint
curl -X POST "http://localhost:8000/api/v1/new-service/" \
  -H "Content-Type: application/json" \
  -d '{"data": "test"}'
```

## Running the Application

### Development Mode

```bash
# With auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or simply
python main.py
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_deskew.py

# Run with coverage
pytest --cov=services --cov=main
```

### API Documentation

FastAPI automatically generates interactive docs:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## Current Services

### Deskew Service

**Endpoint**: `POST /api/v1/deskew/`

**Description**: Corrects skew and perspective in document images.

**Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/deskew/" \
  -F "image=@document.jpg" \
  -F "return_angle=true"
```

**Response**:
```json
{
  "image": "base64_encoded_image...",
  "detected_angle": 2.5,
  "corrected": true,
  "format": "jpeg",
  "processing_time_ms": 1234.56
}
```

## Benefits of This Approach

1. **Simple**: Each service is just one file
2. **Modular**: Services are independent and can be added/removed easily
3. **Familiar**: Similar to NestJS module structure
4. **Testable**: Easy to test individual services
5. **Scalable**: Can split into separate microservices later if needed
6. **Fast Development**: No complex folder structures or boilerplate

## Migration Path

If a service grows too large, you can easily split it:

```
services/
├── deskew_service.py          # Simple service (single file)
└── complex_service/           # Complex service (folder)
    ├── __init__.py
    ├── router.py              # Exports router
    ├── models.py
    ├── logic.py
    └── utils.py
```

Then import it the same way:
```python
from services.complex_service import router as complex_router
app.include_router(complex_router, prefix="/api/v1")
```

## Comparison with Original Design

### Before (Complex)
```
services/deskew/
├── app/
│   ├── core/
│   │   └── deskew_engine.py
│   ├── config.py
│   ├── models.py
│   └── main.py
├── tests/
│   ├── unit/
│   └── integration/
└── requirements.txt
```

### After (Simple)
```
services/
└── deskew_service.py          # Everything in one file!
```

## When to Use Each Approach

**Use Simple (Current)**:
- Starting new project
- Services are < 500 lines
- Fast iteration needed
- Team prefers simplicity

**Use Complex (Original)**:
- Service > 1000 lines
- Multiple developers per service
- Need strict separation of concerns
- Enterprise requirements

## Next Steps

1. Add more services as single files in `services/`
2. When a service gets large (>500 lines), consider splitting it
3. Add shared utilities in `services/shared.py` if needed
4. Keep tests simple and focused
