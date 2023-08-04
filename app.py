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
    potential_models = request.args.getlist("model")

    selected_models = set(potential_models) & set(legal_models)

    if not selected_models:
        selected_models = legal_models

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


@app.route("/precision_recall_f1")
def show_precision_f1_recall():
    potential_models = request.args.getlist("model")
    selected_models = potential_models and legal_models

    if not selected_models:
        selected_models = legal_models

    k_value = request.args.get("k")

    if not k_value:
        k_value = 10
    else:
        k_value = int(k_value)

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


@app.route("/api/v1/mae_rmse/<model_name>", methods=["GET"])
def get_mae_rmse_model(model_name):
    data = Dataset.load_builtin("ml-100k")
    reader = Reader(rating_scale=(1, 5))

    train_data, test_data = train_test_split(data, train_size=0.8)
    model = utils.get_model_instance(model_name)
    start_time = time.time()
    model.fit(train_data)

    predictions = model.test(test_data)
    end_time = time.time()
    mae_score = mae(predictions, verbose=False)
    rmse_score = rmse(predictions, verbose=False)

    return jsonify(rmse_score, mae_score, end_time - start_time)


@app.route("/api/v1/precision_recall_f1/<model_name>/<k>", methods=["GET"])
def get_recall_precision_f1_model_k(model_name, k):
    data = Dataset.load_builtin("ml-100k")
    reader = Reader(rating_scale=(1, 5))

    train_data, test_data = train_test_split(data, train_size=0.8)
    
    model = utils.get_model_instance(model_name)

    model.fit(train_data)

    predictions = model.test(test_data)

    k = int(k)

    precision_score, recall_score, f1_score = eval_func.precision_recall_f1(
        test_data, predictions, k
    )

    return jsonify(precision_score, recall_score, f1_score)

@app.route("/api/v1/robustness/<model_name>/<nr_users>/<rating>/", methods=["GET"])
def get_robustness(model_name, nr_users, rating):
    
    try:
        nr_users = int(nr_users)
        rating = float(rating)
    except Exception as e:
        return jsonify({"Error":str(e)}), 500
    
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_builtin("ml-100k")
    data = data.build_full_trainset()
    data = pd.DataFrame(data.all_ratings())
    
    if rating > data[2].max() or rating < data[2].min():
        return jsonify({"Error":f"rating needs to be in the dataset's ratings interval: [{data[2].min()}:{data[2].max()}]"})
    
    item_column = data.columns.tolist()[1]
    items_list = data.sort_values(by=item_column)[item_column].unique()
    
    def create_user(nr_votes, rating):
        user_id = int(data[0].max() + 1)
        user_rated_items = items_list[:nr_votes]

        user_ids = [user_id] * nr_votes

        user_ratings = pd.DataFrame(columns=data.columns)
        user_ratings[0] = user_ids
        user_ratings[1] = user_rated_items
        user_ratings[2] = rating

        return user_ratings
    
    for _ in range(nr_users):
        data = pd.concat([data,create_user(len(items_list),rating)], ignore_index=True)
        
    data = Dataset.load_from_df(data, reader=reader)
    train_data, test_data = train_test_split(data, train_size = 0.8)
    
    model = utils.get_model_instance(model_name)
    
    model.fit(train_data)
    
    predictions = model.test(test_data)
    
    mae_score = mae(predictions, verbose=False)
    rmse_score = rmse(predictions, verbose=False)
    
    return jsonify(mae_score, rmse_score, nr_users, rating)


if __name__ == "__main__":
    app.run(port=8000, debug=True)
