from flask import jsonify, Flask, request
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae
from plotly import graph_objects as go

import time
import utils
import eval_functions as eval_func
import pandas as pd

app = Flask(__name__)


@app.route("/api/v1/mae_rmse", methods=["POST"])
def post_mae_rmse_model():
    request_data = request.json

    model_name = request_data.get("model_name")
    random_state_value = int(request_data.get("random_state_value"))

    data = Dataset.load_builtin("ml-100k")
    train_data, test_data = train_test_split(
        data, train_size=0.8, random_state=random_state_value
    )

    model = utils.get_model_instance(model_name)

    start_time = time.time()
    model.fit(train_data)

    predictions = model.test(test_data)
    end_time = time.time()

    mae_score = mae(predictions, verbose=False)
    rmse_score = rmse(predictions, verbose=False)

    response_data = {
        "rmse_score": rmse_score,
        "mae_score": mae_score,
        "execution_time": end_time - start_time,
    }

    return jsonify(response_data)


@app.route("/api/v1/precision_recall_f1", methods=["POST"])
def post_precision_recall_f1_model_k():
    request_data = request.json

    model_name = request_data.get("model_name")
    random_state_value = int(request_data.get("random_state_value"))
    k = int(request_data.get("k"))

    data = Dataset.load_builtin("ml-100k")
    train_data, test_data = train_test_split(
        data, train_size=0.8, random_state=random_state_value
    )

    model = utils.get_model_instance(model_name)
    model.fit(train_data)

    predictions = model.test(test_data)

    precision_score, recall_score, f1_score = eval_func.precision_recall_f1(
        test_data, predictions, k
    )

    response_data = {
        "precision_score": precision_score,
        "recall_score": recall_score,
        "f1_score": f1_score,
    }

    return jsonify(response_data)


@app.route("/api/v1/robustness", methods=["POST"])
def post_robustness():
    try:
        request_data = request.json
        model_name = request_data.get("model_name")
        nr_users = int(request_data.get("nr_users"))
        rating = float(request_data.get("rating"))
        random_state_value = int(request_data.get("random_state_value"))
        comparison_method = request_data.get("comparison_method")
        k_value = request_data.get("k_value")

        if comparison_method == "prf" and k_value is None:
            raise ValueError(
                "K needs to be specified for the precision, recall and f1 metrics"
            )

    except Exception as e:
        return jsonify({"Error": str(e)}), 400

    if comparison_method == "prf" and k_value is None:
        return (
            jsonify(
                {
                    "Error": f"K needs to be specified for the precision, recall and f1 metrics"
                }
            ),
            400,
        )

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_builtin("ml-100k")

    train_data, test_data = train_test_split(
        data, train_size=0.8, random_state=random_state_value
    )

    train_data = pd.DataFrame(train_data.build_testset())
    data = pd.DataFrame(data.build_full_trainset().all_ratings())

    if rating > data[2].max() or rating < data[2].min():
        return jsonify(
            {
                "Error": f"rating needs to be in the dataset's ratings interval: [{data[2].min()}:{data[2].max()}]"
            }
        )

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
        train_data = pd.concat(
            [train_data, create_user(len(items_list), rating)], ignore_index=True
        )

    train_data = Dataset.load_from_df(train_data, reader=reader)
    train_data = train_data.build_full_trainset()

    model = utils.get_model_instance(model_name)

    model.fit(train_data)

    predictions = model.test(test_data)

    response_body = {"rating": rating, "nr_users": nr_users}

    if comparison_method == "mae_rmse":
        mae_score = mae(predictions, verbose=False)
        rmse_score = rmse(predictions, verbose=False)

        response_body["mae_score"] = mae_score
        response_body["rmse_score"] = rmse_score

    elif comparison_method == "prf":
        precision_score, recall_score, f1_score = eval_func.precision_recall_f1(
            test_data, predictions, k_value
        )

        response_body["precision_score"] = precision_score
        response_body["recall_score"] = recall_score
        response_body["f1_score"] = f1_score

    return jsonify(response_body)


if __name__ == "__main__":
    app.run(port=8000, debug=True, threaded=True)
