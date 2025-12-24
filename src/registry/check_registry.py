from mlflow.tracking import MlflowClient

client = MlflowClient()

models = client.search_registered_models()
for m in models:
    print("Model:", m.name)
    for v in m.latest_versions:
        print("  Version:", v.version, "Stage:", v.current_stage)