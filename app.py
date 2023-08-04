from flask import Flask, render_template, request, jsonify
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae
import plotly.graph_objects as go

import time
import utils
import eval_functions as eval_func
import pandas as pd

app = Flask(__name__)

# Load the dataset
legal_models = ["SVD", "KNN", "Baseline", "SlopeOne", "CoClustering"]

@app.route("/")
def show_main():
    return render_template("main.html")


@app.route("/mae_rmse")
def show_mae_rmse_graph():
    selected_models = utils.get_models()

    mae_to_plot, rmse_to_plot, time_to_plot = utils.get_plots(
        selected_models, utils.get_mae_rmse_score_bars
    )

    fig_mae = go.Figure(data=mae_to_plot)
    fig_mae.update_layout(title="Eroarea medie absoluta (MAE)", barmode="group")

    fig_rmse = go.Figure(data=rmse_to_plot)
    fig_rmse.update_layout(
        title="Radacina erorii mediei patratice (RMSE)", barmode="group"
    )

    fig_time = go.Figure(data=time_to_plot)
    fig_time.update_layout(
        title="Timpul de antrenare si de predictie a modelelor",
        barmode="group",
        yaxis=dict(
            range=(0, max(time_to_plot, key=lambda time: time["y"])["y"][0] + 0.2)
        ),
    )

    return render_template(
        "mae_rmse.html",
        mae_plot=fig_mae.to_html(),
        rmse_plot=fig_rmse.to_html(),
        time_plot=fig_time.to_html(),
    )

@app.route("/robustness")
def show_robustness_graph():
    selected_models = utils.get_models()


@app.route("/precision_recall_f1")
def show_precision_f1_recall():
    selected_models = utils.get_models()

    k_value = utils.get_k_value()

    precision_to_plot, recall_to_plot, f1_to_plot = utils.get_plots(
        selected_models, utils.get_precision_recall_f1_score_bars, k_value
    )

    fig_precision = go.Figure(data=precision_to_plot)
    fig_precision.update_layout(title="Precision", barmode="group")

    fig_recall = go.Figure(data=recall_to_plot)
    fig_recall.update_layout(title="Recall", barmode="group")

    fig_f1 = go.Figure(data=f1_to_plot)
    fig_f1.update_layout(title="F1", barmode="group")

    return render_template(
        "precision_recall_f1.html",
        precision_plot=fig_precision.to_html(),
        recall_plot=fig_recall.to_html(),
        f1_plot=fig_f1.to_html(),
    )


if __name__ == "__main__":
    app.run(port=8080, debug=True)
