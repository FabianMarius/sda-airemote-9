import mlflow

# Set or create an experiment
mlflow.set_experiment("My New Experiment")

# Start a new run within the experiment
with mlflow.start_run():
    # Log parameters, metrics, and artifacts
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_artifact("E:\\SDA\\Courses\\AIRemote\\done_AIRemoteRO9\\done_before\\deep_learning (16)\\deep_learning\\model.h5")