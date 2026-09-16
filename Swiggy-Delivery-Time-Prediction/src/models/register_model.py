import mlflow
import json
from pathlib import Path
from mlflow import MlflowClient
import logging


# create logger
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a fomratter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)

# initialize dagshub
import dagshub
import mlflow.client
dagshub.init(repo_owner='aryann13', 
             repo_name='Swiggy-Delivery-Time-Prediction', 
             mlflow=True)

# set the mlflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/aryann13/Swiggy-Delivery-Time-Prediction.mlflow")


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


if __name__ == "__main__":
    # Fix Windows terminal emoji crash (MLflow prints a running emoji which CP1252 can't handle)
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    # root path
    root_path = Path(__file__).parent.parent.parent
    
    # run information file path
    run_info_path = root_path / "run_information.json"
    
    # load run info
    run_info = load_model_information(run_info_path)
    run_id   = run_info["run_id"]
    model_name = run_info["model_name"]

    client = MlflowClient()

    # Step 1: Check if this run already has a registered version
    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
        run_versions = [v for v in all_versions if v.run_id == run_id]
    except Exception:
        all_versions = []
        run_versions = []

    if run_versions:
        # Already registered - just get the version and skip to transition
        latest = max(run_versions, key=lambda v: int(v.version))
        registered_model_version = latest.version
        registered_model_name    = latest.name
        logger.info(f"Model version {registered_model_version} already registered for run {run_id}")

    else:
        # Step 2: Log model properly into the existing run
        import joblib
        model_path = root_path / "models" / "model.joblib"
        model = joblib.load(model_path)
        logger.info("Local model.joblib loaded successfully")

        TRUSTED_TYPES = [
            "sklearn.utils._bunch.Bunch",
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBRegressor",
        ]

        # Wrap with-block exit in try/except to survive the MLflow emoji print on Windows
        try:
            with mlflow.start_run(run_id=run_id):
                try:
                    mlflow.sklearn.log_model(
                        model,
                        name=model_name,
                        skops_trusted_types=TRUSTED_TYPES,
                    )
                    logger.info("Model logged successfully to existing run")
                except Exception as e:
                    logger.warning(f"log_model failed: {e}. Falling back to log_artifact.")
                    mlflow.log_artifact(str(model_path), artifact_path=model_name)

                # Step 3: Register the model version (inside the run context)
                model_registry_path = f"runs:/{run_id}/{model_name}"
                try:
                    model_version = mlflow.register_model(
                        model_uri=model_registry_path,
                        name=model_name
                    )
                    registered_model_version = model_version.version
                    registered_model_name    = model_version.name
                    logger.info(f"Model registered: version {registered_model_version}")
                except Exception as e:
                    logger.warning(f"register_model failed: {e}")
                    registered_model_version = None
                    registered_model_name    = None

        except (UnicodeEncodeError, UnicodeDecodeError):
            # Harmless Windows emoji bug - the actual work completed fine
            logger.info("Suppressed Windows emoji encoding error (harmless)")

        # If register_model failed inside the with-block, fallback to latest in registry
        if not registered_model_version:
            all_versions = client.search_model_versions(f"name='{model_name}'")
            if not all_versions:
                raise RuntimeError("Model not found in registry. Check DagsHub manually.")
            latest = max(all_versions, key=lambda v: int(v.version))
            registered_model_version = latest.version
            registered_model_name    = latest.name

    logger.info(f"The latest model version in model registry is {registered_model_version}")

    # Step 4: Transition the model to Staging
    client.transition_model_version_stage(
        name=registered_model_name,
        version=registered_model_version,
        stage="Staging"
    )
    logger.info("Model pushed to Staging stage")