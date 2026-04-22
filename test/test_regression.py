import json
from fastapi.testclient import TestClient
from backend import app

client = TestClient(app)

def test_regression():
    file_path = 'house_price_regression_dataset.csv'
    with open(file_path, 'rb') as f:
        response = client.post(
            "/train",
            data={"task_type": "Regression", "target_column": "House_Price"},
            files={"file": ("house_price_regression_dataset.csv", f, "text/csv")}
        )
    print("Status:", response.status_code)
    try:
        print("JSON Output:", json.dumps(response.json(), indent=2))
    except Exception as e:
        print("Text:", response.text)

if __name__ == "__main__":
    test_regression()
