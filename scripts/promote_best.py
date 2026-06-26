import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

# Find run with highest best_val_acc
exp  = client.get_experiment_by_name("focus_classifier")
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["metrics.best_val_acc DESC"],
    max_results=1,
)
best   = runs[0]
run_id = best.info.run_id
acc    = best.data.metrics["best_val_acc"]
print(f"Best run : {run_id}")
print(f"val_acc  : {acc:.4f}")

# Register it
mv = mlflow.register_model(f"runs:/{run_id}/model", "FocusClassifier")
print(f"Registered version {mv.version}")

# Staging first
client.transition_model_version_stage("FocusClassifier", mv.version, "Staging")
print(f"Version {mv.version} -> Staging")

# Then Production
client.transition_model_version_stage("FocusClassifier", mv.version, "Production")
print(f"Version {mv.version} -> Production")