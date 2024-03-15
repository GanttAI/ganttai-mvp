from modal import Stub, Image, asgi_app
import numpy as np
import pandas as pd
import joblib
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from scipy import stats
import os

pd.options.mode.copy_on_write = True

app_location = "./"
os.system(f"ls {app_location}")
os.system(f"pwd")

image = (
    # Image.debian_slim(force_build=True)
    Image.debian_slim()
    .run_commands(
        "pip install -U fastapi pandas numpy joblib pydantic scipy scikit-learn"
    )
    .copy_local_dir("./", "/root/")
)
stub = Stub("client-ganttai-ProjectML-v3")


class DemoInputs(BaseModel):
    location: str
    size: int
    complexity: str
    price: float


model_v3 = joblib.load(os.path.join(app_location, "GantAI_Model_V3.pkl"))

web_app = FastAPI()
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data = pd.read_csv(os.path.join(app_location, "projects_3_with_duration.csv"))
# Select the features needed for prediction
X = data[["size", "complexity", "price", "location"]]

# Convert categorical variables to binary variables
X = pd.get_dummies(X)
y = data["duration"]

predictions = model_v3.predict(X)
residuals = y - predictions


def calculate_task_duration(df_task_ids):
    df = pd.DataFrame({"task_id": df_task_ids})
    # add duration column with random a number between 1 and 1000:
    df["duration"] = df["task_id"].apply(lambda x: np.random.randint(1, 600))
    return df


def A_step(location, size, complexity, price):
    df = pd.read_csv(os.path.join(app_location, "projects_3.csv"))

    # Define the input project

    # Filter the dataset for projects in the same location and with the same complexity

    filtered_df = df[(df["location"] == location)]
    if filtered_df.empty:
        filtered_df = df

    if complexity == "High":
        complexity_val = 1000000
    elif complexity == "Medium":
        complexity_val = 500000
    elif complexity == "Low":
        complexity_val = 100000

    # Calculate the Euclidean distance for each project
    distances = np.sqrt(
        (filtered_df["size"] - size) * 2
        + (filtered_df["price"] - price) * 2
        # + (filtered_df["complexity"] - complexity_val) * 2
        + (filtered_df["complexity"] - complexity_val) * 2
    )

    # Add distances to DataFrame
    filtered_df["distance"] = distances

    # Sort the projects by distance and select the top 3 closest ones
    closest_projects = filtered_df.sort_values(by="distance").head(3)
    return closest_projects


@web_app.post("/run")
def get_task_ids(body: DemoInputs, response_class=StreamingResponse):
    location = body.location
    size = body.size
    complexity = body.complexity
    price = body.price

    closest_projects = A_step(location, size, complexity, price)
    closest_proj = closest_projects.iloc[0][
        "proj_id"
    ]  # TO-DO: Edge cases should be gracefully handled.

    df = pd.read_csv(os.path.join(app_location, "tasks_4_w_name_n_duration.csv"))
    df = df[df["proj_id"] == closest_proj]
    # print(df)

    df = df[["task_id", "task_name", "duration"]]
    return StreamingResponse(
        iter([df.to_csv(index=False)]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=data.csv"},
    )


@web_app.post("/proj-duration")
def get_task_ids(body: DemoInputs):
    location = body.location
    size = body.size
    complexity = body.complexity
    price = body.price
    print(location)
    if complexity == "High":
        complexity_val = 1000000
    elif complexity == "Medium":
        complexity_val = 500000
    elif complexity == "Low":
        complexity_val = 100000
    complexity = complexity_val

    dic = {
        "size": [size],
        "complexity": [complexity],
        "price": [price],
        "location": [location],
    }

    df = pd.DataFrame(dic)
    df = pd.get_dummies(df)
    df = df.align(X, join="outer", axis=1, fill_value=0)[0]
    ordered_dict = {
        "size": df["size"],
        "complexity": df["complexity"],
        "price": df["price"],
        "location_Arbor-Lyn": df["location_Arbor-Lyn"],
        "location_Cardinal Grove": df["location_Cardinal Grove"],
        "location_Durham Farms": df["location_Durham Farms"],
        "location_Forest Hills": df["location_Forest Hills"],
        "location_Governors": df["location_Governors"],
        "location_Hailey's Glen": df["location_Hailey's Glen"],
        "location_Marlin Chase": df["location_Marlin Chase"],
        "location_Marsh Island": df["location_Marsh Island"],
        "location_Mosaic": df["location_Mosaic"],
        "location_NewMarket": df["location_NewMarket"],
        "location_Outer Banks": df["location_Outer Banks"],
        "location_Peninsula": df["location_Peninsula"],
        "location_Peninsula Lakes": df["location_Peninsula Lakes"],
        "location_River Mill": df["location_River Mill"],
        "location_Walden": df["location_Walden"],
        "location_Welches Pond": df["location_Welches Pond"],
    }
    ordered_df = pd.DataFrame(ordered_dict)
    # print(ordered_df)
    value = model_v3.predict(ordered_df)
    new_input = ordered_df
    MSE = np.mean(residuals**2)
    n = len(X)
    p = X.shape[1]  # Number of predictors
    X_mean = np.mean(X, axis=0)
    squared_differences = (new_input - X_mean) ** 2

    # Sum across predictors (axis=1 if new_input is 2D, else just sum directly)
    if new_input.ndim > 1:
        sum_squared_differences = np.sum(squared_differences, axis=1)
    else:
        sum_squared_differences = np.sum(squared_differences)

    # Now use this in the SE calculation
    # standard_error = np.sqrt(
    #     MSE * (1 + 1 / n + sum_squared_differences / np.sum((X - X_mean) ** 2))
    # )

    standard_error = np.sqrt(
        MSE
        * (
            1 / n
            + (new_input - np.mean(X, axis=0)) ** 2
            / np.sum((X - np.mean(X, axis=0)) ** 2, axis=0)
        )
    )

    # Define confidence level
    confidence = 0.95
    t_value = stats.t.ppf((1 + confidence) / 2, df=n - p - 1)

    # Calculate confidence interval for new prediction
    new_prediction = value[0]
    interval = t_value * standard_error

    lower_bound = new_prediction - interval
    upper_bound = new_prediction + interval
    diff = upper_bound - lower_bound

    features = ["size", "complexity", "price"]
    print("Predicted duration:\n", new_prediction)
    print("Confidence interval:\n", lower_bound[features], upper_bound[features])
    print("Difference:\n", diff[features])

    # example diff:
    # size          165.764929
    # complexity    183.015102
    # price         187.053645

    # I want to get the mean of the differences

    diff_mean = diff[features].mean()
    # print(diff_mean)
    plus_minus_diff = (diff_mean / 2).mean()
    # print(value)
    print("Mean of the differences (diff_mean / 2):", int(plus_minus_diff))
    return {"duration": int(value[0]), "confidence": int(plus_minus_diff)}


@web_app.get("/web-ui/")
def read_root():
    return FileResponse(os.path.join(app_location, "front-app.html"))


@stub.function(image=image, concurrency_limit=100, keep_warm=0)
@asgi_app()
def fastapi_app():
    return web_app
