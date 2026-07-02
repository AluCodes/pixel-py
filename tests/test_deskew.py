"""
Tests for deskew service.
"""
import pytest
from fastapi.testclient import TestClient
from main import app
import io
import numpy as np
import cv2

client = TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create a simple test image."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (550, 350), (0, 0, 0), 2)
    cv2.putText(img, "TEST", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    success, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()


def test_deskew_health():
    """Test deskew health endpoint."""
    response = client.get("/api/v1/deskew/health")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "deskew"
    assert data["status"] == "healthy"


def test_deskew_with_valid_image(sample_image_bytes):
    """Test deskewing a valid image."""
    files = {"image": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
    response = client.post("/api/v1/deskew/", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "image" in data
    assert "detected_angle" in data
    assert "corrected" in data
    assert "format" in data
    assert "processing_time_ms" in data
    
    assert isinstance(data["image"], str)
    assert isinstance(data["detected_angle"], (int, float))
    assert isinstance(data["corrected"], bool)
    assert data["format"] == "jpeg"


def test_deskew_without_image():
    """Test deskew endpoint without providing an image."""
    response = client.post("/api/v1/deskew/")
    assert response.status_code == 422  # Unprocessable Entity


def test_deskew_with_invalid_image():
    """Test deskew with invalid image data."""
    files = {"image": ("test.jpg", io.BytesIO(b"not an image"), "image/jpeg")}
    response = client.post("/api/v1/deskew/", files=files)
    assert response.status_code == 400
