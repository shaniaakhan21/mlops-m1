# Iris Flower Classification Model

This project implements a machine learning model to classify iris flowers into three species based on their petal and sepal dimensions using the Iris dataset. The model is built with scikit-learn and exposed as a simple Flask web application.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/shaniaakhan21/mlops-m1.git
    ```

2. Install dependencies:
    ```bash
    cd mlops-m1
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    python app.py
    ```

The Flask app will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Usage

### API Endpoint

- **POST /predict**
    - **Request**: 
      ```json
      {
        "features": [5.1, 3.5, 1.4, 0.2]
      }
      ```
    - **Response**: 
      ```json
      {
        "prediction": 0
      }
      ```
    - `0`: Setosa, `1`: Versicolor, `2`: Virginica

## Model Details

- **Algorithm**: K-Nearest Neighbors (KNN)
- **Input Features**: Sepal Length, Sepal Width, Petal Length, Petal Width
- **Output**: Iris flower species

## License

MIT License

