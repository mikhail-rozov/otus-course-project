import os
import pickle

import boto3
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import subprocess


PROJECT_DIR = "/home/airflow"
MLFLOW_TRACKING_SERVER = "10.0.0.17"
BUCKET_NAME = "mrozov-mlops"
BUCKET_PATH = BUCKET_NAME + "/" + "course_project/data/"
PARQUET_FILE_NAME_TRAIN = "train_dataset.parquet"
PARQUET_FILE_NAME_TEST = "test_dataset.parquet"
RANDOM_STATE = 29


def git_commit():
    subprocess.run(["git", "-C", PROJECT_DIR, "add", "model.pkl"], check=True)
    subprocess.run(["git", "-C", PROJECT_DIR, "commit", "-m", "new model version"], check=True)
    subprocess.run(["git", "-C", PROJECT_DIR, "push", f"https://{os.getenv("GITHUB_TOKEN")}@github.com/mikhail-rozov/otus-course-project.git"], check=True)


def main():
    mlflow.set_tracking_uri(f"http://{MLFLOW_TRACKING_SERVER}:5000")
    client = MlflowClient()

    boto3_session = boto3.session.Session()
    s3 = boto3_session.client(
        service_name="s3",
        endpoint_url="https://storage.yandexcloud.net",
        region_name="ru-central1"
    )

    df_train = pd.read_parquet("s3://" + BUCKET_PATH + PARQUET_FILE_NAME_TRAIN, 
                           engine="pyarrow", 
                           storage_options={
                             "client_kwargs": {
                                 "endpoint_url": "https://storage.yandexcloud.net",
                                 "region_name": "ru-central1"}
                           })

    df_test = pd.read_parquet("s3://" + BUCKET_PATH + PARQUET_FILE_NAME_TEST, 
                            engine="pyarrow", 
                            storage_options={
                                "client_kwargs": {
                                    "endpoint_url": "https://storage.yandexcloud.net",
                                    "region_name": "ru-central1"}
                            })
    
    df_train.drop(["ip", "click_time"], axis=1, inplace=True)
    df_test.drop(["ip", "click_time"], axis=1, inplace=True)
    df_train.dropna(inplace=True)

    experiment = mlflow.set_experiment("project_experiment")
    experiment_id = experiment.experiment_id

    if len(client.search_runs(experiment_id, max_results=1)) < 1:
        is_first = True
    else:
        is_first = False
        best_run = client.search_runs(experiment_id, order_by=["metrics.roc_auc_mean DESC"], max_results=1)[0]

    with mlflow.start_run(experiment_id=experiment_id):
        # model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        model = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
        model.fit(df_train.drop("is_attributed", axis=1), df_train["is_attributed"])

        preds = model.predict_proba(df_test.drop("is_attributed", axis=1))[:, 1]
        df_test["prediction"] = preds
        
        current_metrics = []
        N_SAMPLES = 100

        for _ in range(N_SAMPLES):
            sample = df_test.sample(frac=1.0, replace=True)
            roc_auc = roc_auc_score(sample["is_attributed"], sample["prediction"])
            mlflow.log_metric("roc_auc", roc_auc)
            current_metrics.append(roc_auc)

        roc_auc_mean = sum(current_metrics) / N_SAMPLES
        mlflow.log_metric("roc_auc_mean", roc_auc_mean)

        # If it's the first run then just save the model
        if is_first:
            with open(f"{PROJECT_DIR}/model.pkl", "wb") as f:
                pickle.dump(model, f)
            s3.upload_file(f"{PROJECT_DIR}/model.pkl", BUCKET_NAME, "course_project/models/model.pkl")
            git_commit()
        # If it's not, and the mean is higher than in the best run so far, then do 2-samples independent t-test
        # to check if the change is significant 
        else:
            if roc_auc_mean > best_run.data.metrics.get("roc_auc_mean", 0):
                best_run_id = best_run.info.run_id
                best_metrics = []

                for best_metric in client.get_metric_history(best_run_id, "roc_auc"):
                    best_metrics.append(best_metric.value)

                pvalue = ttest_ind(best_metrics, current_metrics).pvalue
                mlflow.log_metric("p-value", pvalue)

                # If the new mean is significantly higher than the previous one, save the model
                alpha = 0.05
                if pvalue < alpha:
                    with open(f"{PROJECT_DIR}/model.pkl", "wb") as f:
                        pickle.dump(model, f)
                    s3.upload_file(f"{PROJECT_DIR}/model.pkl", BUCKET_NAME, "course_project/models/model.pkl")
                    git_commit()

if __name__ == "__main__":
    main()
