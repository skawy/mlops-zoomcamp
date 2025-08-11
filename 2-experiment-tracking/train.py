import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import mlflow
# import mlflow.sklearn




mlflow.set_tracking_uri("sqlite:///mlflow2.db")
mlflow.set_experiment("homework-experiment")



def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


# @click.command()
# @click.option(
#     "--data_path",
#     default="./output",
#     help="Location where the processed NYC taxi trip data was saved"
# )




data_path = "output"

dv = load_pickle(os.path.join(data_path, "dv.pkl"))
# mlflow.sklearn.autolog()

with mlflow.start_run() as run:

    mlflow.set_tag("developer", "skawy")
   
    max_depth = 10
    random_state = 0

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    
    rf = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
    rf.fit(X_train, y_train)

    print(rf.get_params())
    y_pred = rf.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred)
    mlflow.log_metric("rmse", rmse)

    with open("models/preprocessor.b", "wb") as f_out:
        pickle.dump((dv,rf), f_out)
    mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")


    mlflow.sklearn.log_model(
        sk_model=rf,
        name="rf-model",
        registered_model_name="sk-learn-random-forest-reg-model",
    )


# import os
# import pickle
# import click

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import root_mean_squared_error
# import mlflow
# import mlflow.sklearn






# from sklearn.datasets import make_regression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split

# import mlflow
# import mlflow.sklearn

# mlflow.set_tracking_uri("sqlite:///mlflow2.db")
# mlflow.set_experiment("homework-experiment")

# mlflow.sklearn.autolog()

# with mlflow.start_run() :
#     X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     params = {"max_depth": 2, "random_state": 42}
#     model = RandomForestRegressor(**params)
#     model.fit(X_train, y_train)

#     # Log parameters and metrics using the MLflow APIs
#     # mlflow.log_params(params)

#     y_pred = model.predict(X_test)
#     # mlflow.log_metrics({"mse": mean_squared_error(y_test, y_pred)})

#     # Log the sklearn model and register as version 1
#     # mlflow.sklearn.log_model(
#     #     sk_model=model,
#     #     name="sklearn-model",
#     #     input_example=X_train,
#     #     registered_model_name="sk-learn-random-forest-reg-model",
#     # )