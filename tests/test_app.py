import pytest
from app import app  # Import your Flask app
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import os
from app import app


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_predict(client):
    # Send a POST request to the /predict endpoint with sample data
    data = {"features": [5.1, 3.5, 1.4, 0.2]}  # Example data for Iris setosa
    response = client.post('/predict', json=data)
    
    # Assert that the response status code is 200
    assert response.status_code == 200
    
    # Assert that the response contains a prediction key
    json_data = response.get_json()
    assert 'prediction' in json_data

