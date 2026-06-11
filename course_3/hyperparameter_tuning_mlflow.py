from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from pathlib import Path
# setez output-ul pt mllflow in directorul in care se aflta acest script (calea este calculata dinamic)
mlflow.set_tracking_uri(Path(__file__).parent / "mlruns")


# loading the mnist dataset into the notebook
mnist = fetch_openml('mnist_784',version=1)
mlflow.set_experiment("Experiment - Hyerparameter Tuning")

X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, 
                                                    test_size=0.2, 
                                                    random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20]
}
grid = ParameterGrid(param_grid)

for params in grid:
    with mlflow.start_run():
        print("Running with params:", params)
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        # mlflow.log_artifact("model.h5", model)huper