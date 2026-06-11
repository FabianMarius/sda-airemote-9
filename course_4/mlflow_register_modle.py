import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
# setez output-ul pt mllflow in directorul in care se aflta acest script (calea este calculata dinamic)
mlflow.set_tracking_uri(Path(__file__).parent / "mlruns")
# Register the model
autolog_experiment = mlflow.get_experiment_by_name("Experiment - Autolog")

runs_df = mlflow.search_runs(experiment_ids=[autolog_experiment.experiment_id])
run_id = runs_df.iloc[0]["run_id"]
print("run_id:", run_id)

result = mlflow.register_model(f"runs:/{run_id}", "ModelFromAutologExp")

# Transition the model to Production
client = MlflowClient()
client.transition_model_version_stage(
    name="ModelFromAutologExp",
    version=result.version,
    stage="Production"
)