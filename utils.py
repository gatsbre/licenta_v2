import requests
import plotly.graph_objects as go
from surprise import SVD, KNNBasic, BaselineOnly, SlopeOne, CoClustering
import random
from flask import request


def get_model_instance(model_name):
    model_factories = {
        "svd": lambda: SVD(),
        "knn": lambda: KNNBasic(),
        "baseline": lambda: BaselineOnly(),
        "slope_one": lambda: SlopeOne(),
        "co_clustering": lambda: CoClustering(),
    }

    return model_factories[model_name]()


def get_plots(selected_models, function, k=0, nr_users=0, rating=0):
    plots = []

    random_state_value = random.randint(
        0, 2**32 - 1
    )  # the interval of the random_state

    if k:
        for model_name in selected_models:
            for i, score_bar in enumerate(function(model_name, random_state_value, k)):
                try:
                    if not plots[i]:
                        plots[i] = []
                except IndexError:
                    plots.append([])
                plots[i].append(score_bar)

    elif nr_users and rating:
        for model_name in selected_models:
            for i, score_bar in enumerate(
                function(model_name, random_state_value, nr_users, rating)
            ):
                try:
                    if not plots[i % 2]:
                        plots[i % 2] = []
                except IndexError:
                    plots.append([])
                plots[i % 2].append(score_bar)
    else:
        for model_name in selected_models:
            for i, score_bar in enumerate(function(model_name, random_state_value)):
                try:
                    if not plots[i]:
                        plots[i] = []
                except IndexError:
                    plots.append([])
                plots[i].append(score_bar)

    return plots


def get_mae_rmse_score_bars(model_name, random_state_value):
    bar_width = 0.1

    post_data = {"model_name": model_name, "random_state_value": random_state_value}

    response = requests.post("http://127.0.0.1:8000/api/v1/mae_rmse", json=post_data)
    response_data = response.json()

    mae_score = response_data["mae_score"]
    rmse_score = response_data["rmse_score"]
    elapsed_time = response_data["execution_time"]

    mae_bar = go.Bar(
        name=f"{model_name}",
        x=["MAE"],
        y=[mae_score],
        width=bar_width,
        text=[f"{model_name} <br> MAE: {round(mae_score, 6)}"],
        textposition="inside",
        insidetextanchor="middle",
        hoverinfo="text",
        hovertext=f"Model: {model_name} <br>MAE: {mae_score}",
    )
    rmse_bar = go.Bar(
        name=f"{model_name}",
        x=["RMSE"],
        y=[rmse_score],
        width=bar_width,
        text=[f"{model_name} <br> RMSE: {round(rmse_score, 6)}"],
        textposition="inside",
        insidetextanchor="middle",
        hoverinfo="text",
        hovertext=f"Model: {model_name} <br>RMSE: {rmse_score}",
    )
    time_bar = go.Bar(
        name=f"{model_name}",
        x=["Time"],
        y=[elapsed_time],
        width=bar_width,
        text=[f"{model_name} <br> Time: {round(elapsed_time, 6)}s"],
        textposition="outside",
        hoverinfo="text",
        hovertext=f"Model: {model_name} <br>Time: {round(elapsed_time, 6)}s",
    )

    return mae_bar, rmse_bar, time_bar


def get_precision_recall_f1_score_bars(model_name, random_state_value, k):
    bar_width = 0.1

    post_data = {
        "model_name": model_name,
        "random_state_value": random_state_value,
        "k": k,
    }

    response = requests.post(
        "http://127.0.0.1:8000/api/v1/precision_recall_f1", json=post_data
    )
    response_data = response.json()

    precision_score = response_data["precision_score"]
    recall_score = response_data["recall_score"]
    f1_score = response_data["f1_score"]

    precision_bar = go.Bar(
        name=f"{model_name}",
        x=["Precision"],
        y=[precision_score],
        width=bar_width,
        text=[f"{model_name} <br> Precision: {round(precision_score, 6)}"],
        textposition="outside",
        # insidetextanchor="middle",
        hoverinfo="text",
        hovertext=f"Model: {model_name} <br>Precision: {precision_score}",
    )
    recall_bar = go.Bar(
        name=f"{model_name}",
        x=["Recall"],
        y=[recall_score],
        width=bar_width,
        text=[f"{model_name} <br> Recall: {round(recall_score, 6)}"],
        textposition="outside",
        # insidetextanchor="middle",
        hoverinfo="text",
        hovertext=f"Model: {model_name} <br>Recall: {recall_score}",
    )
    f1_bar = go.Bar(
        name=f"{model_name}",
        x=["F1"],
        y=[f1_score],
        width=bar_width,
        text=[f"{model_name} <br> F1: {round(f1_score, 6)}"],
        textposition="outside",
        # insidetextanchor="middle",
        hoverinfo="text",
        hovertext=f"Model: {model_name} <br>F1: {round(f1_score, 6)}",
    )

    return precision_bar, recall_bar, f1_bar


def get_robustness_score_bars(model_name, random_state_value, nr_users, rating):
    bar_width = 0.1

    robustness_post_data = {
        "model_name": model_name,
        "random_state_value": random_state_value,
        "nr_users": nr_users,
        "rating": rating,
    }

    base_post_data = {
        "model_name": model_name,
        "random_state_value": random_state_value,
    }

    response = requests.post(
        "http://127.0.0.1:8000/api/v1/robustness", json=robustness_post_data
    )
    robustness_response_data = response.json()

    response = requests.post(
        "http://localhost:8000/api/v1/mae_rmse", json=base_post_data
    )
    base_response_data = response.json()

    robustness_mae_score = robustness_response_data["mae_score"]
    robustness_rmse_score = robustness_response_data["rmse_score"]
    base_mae_score = base_response_data["mae_score"]
    base_rmse_score = base_response_data["rmse_score"]

    robustness_mae_bar = go.Bar(
        name=f"{model_name}",
        x=["MAE"],
        y=[robustness_mae_score],
        width=bar_width,
        text=[
            f"{model_name} <br>After adding users <br> MAE: {round(robustness_mae_score, 6)}"
        ],
        textposition="inside",
        insidetextanchor="middle",
        hoverinfo="text",
        hovertext=f"Model: {model_name} <br>After adding users <br>MAE: {robustness_mae_score}",
    )
    robustness_rmse_bar = go.Bar(
        name=f"{model_name}",
        x=["RMSE"],
        y=[robustness_rmse_score],
        width=bar_width,
        text=[
            f"{model_name} <br>After adding users <br> RMSE: {round(robustness_rmse_score, 6)}"
        ],
        textposition="inside",
        insidetextanchor="middle",
        hoverinfo="text",
        hovertext=f"Model: {model_name} <br>After adding users <br>RMSE: {robustness_rmse_score}",
    )

    base_mae_bar = go.Bar(
        name=f"{model_name}",
        x=["MAE"],
        y=[base_mae_score],
        width=bar_width,
        text=[
            f"{model_name} <br>Before adding users <br> Base MAE: {round(base_mae_score, 6)}"
        ],
        textposition="inside",
        insidetextanchor="middle",
        hoverinfo="text",
        hovertext=f"Model: {model_name} <br>Before adding users <br>MAE: {base_mae_score}",
    )
    base_rmse_bar = go.Bar(
        name=f"{model_name}",
        x=["RMSE"],
        y=[base_rmse_score],
        width=bar_width,
        text=[
            f"{model_name} <br>Before adding users <br> Base RMSE: {round(base_rmse_score, 6)}"
        ],
        textposition="inside",
        insidetextanchor="middle",
        hoverinfo="text",
        hovertext=f"Model: {model_name} <br>Before adding users <br>RMSE: {base_rmse_score}",
    )

    return robustness_mae_bar, robustness_rmse_bar, base_mae_bar, base_rmse_bar


def get_top_k_percent(sorted_list, k):
    if not (0 <= k <= 100):
        raise ValueError("K should be a value between 0 and 100 (inclusive).")

    k_percent_index = int(len(sorted_list) * (k / 100))
    top_k_percent = sorted_list[:k_percent_index]

    return top_k_percent


def get_models():
    legal_models = ["svd", "knn", "baseline", "slope_one", "co_clustering"]

    potential_models = request.args.getlist("model")

    selected_models = set(potential_models) & set(legal_models)

    if not selected_models:
        selected_models = legal_models

    return selected_models


def get_k_value():
    k_value = request.args.get("k")

    if not k_value:
        k_value = 10
    else:
        k_value = int(k_value)

    return k_value


def get_comparison_method():
    legal_methods = ["mae_rmse", "precision_recall_f1"]

    comparison_method = request.args.get("comparison_method")

    if comparison_method not in legal_methods:
        comparison_method = legal_methods[0]

    return comparison_method
