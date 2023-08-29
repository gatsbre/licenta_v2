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
    try:
        request_data = request.json

        model_name = request_data.get("model_name")
        random_state_value = int(request_data.get("random_state_value"))
        selected_dataset = request_data.get("dataset")

    except Exception as exception:
        return jsonify({"Error": str(exception)}), 400

    model = utils.get_model_instance(model_name)
    selected_dataset = utils.get_dataset(selected_dataset)

    data = Dataset.load_builtin(selected_dataset)
    train_data, test_data = train_test_split(
        data, train_size=0.8, random_state=random_state_value
    )

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
    try:
        request_data = request.json

        model_name = request_data.get("model_name")
        random_state_value = int(request_data.get("random_state_value"))
        k = int(request_data.get("k"))
        selected_dataset = request_data.get("dataset")

    except Exception as exception:
        return jsonify({"Error": str(exception)}), 400

    selected_dataset = utils.get_dataset(selected_dataset)

    data = Dataset.load_builtin(selected_dataset)

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


# TODO: de adaugat dataseturi pentru robustness si scarcity
@app.route("/api/v1/robustness", methods=["POST"])
def post_robustness():
    try:
        request_data = request.json

        model_name = request_data.get("model_name")
        nr_users = int(request_data.get("nr_users"))
        rating = float(request_data.get("rating"))
        random_state_value = int(request_data.get("random_state_value"))
        comparison_method = request_data.get("comparison_method")
        k_value = int(request_data.get("k"))
        selected_dataset = request_data.get("dataset")

        if comparison_method is None:
            raise KeyError(
                "comparison_method needs to be specified in the request body"
            )

        if comparison_method == "prf" and k_value is None:
            raise KeyError(
                "k needs to be specified for the precision, recall and f1 metrics"
            )

    except Exception as e:
        return jsonify({"Error": str(e)}), 400

    selected_dataset = utils.get_dataset(selected_dataset)

    data = Dataset.load_builtin(selected_dataset)

    train_data, test_data = train_test_split(
        data, train_size=0.8, random_state=random_state_value
    )

    train_data = pd.DataFrame(train_data.build_testset())
    data = pd.DataFrame(data.build_full_trainset().all_ratings())
    reader = Reader(rating_scale=(data[2].min(), data[2].max()))

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


@app.route("/api/v1/scarcity", methods=["POST"])
def post_scarcity():
    try:
        request_data = request.json

        model_name = request_data.get("model_name")
        random_state_value = int(request_data.get("random_state_value"))
        comparison_method = request_data.get("comparison_method")
        k_value = int(request_data.get("k"))
        model_feeding_rate = int(request_data.get("model_feeding_rate"))

        if comparison_method is None:
            raise KeyError(
                "comparison_method needs to be specified in the request body"
            )

        if comparison_method == "prf" and k_value is None:
            raise KeyError(
                "k needs to be specified for the precision, recall and f1 metrics"
            )
    except Exception as e:
        return jsonify({"Error": str(e)}), 400

    table = []
    data = Dataset.load_builtin("ml-100k")

    trainset, testset = train_test_split(
        data, train_size=0.8, random_state=random_state_value
    )

    model = utils.get_model_instance(model_name)

    trainset_df = pd.DataFrame(trainset.build_testset())

    reader = Reader(rating_scale=(trainset_df[2].min(), trainset_df[2].max()))

    len_trainset_df = len(trainset_df)

    for i in range(
        len_trainset_df // model_feeding_rate,
        len_trainset_df // model_feeding_rate + len_trainset_df,
        len_trainset_df // model_feeding_rate,
    ):
        partial_trainset = Dataset.load_from_df(trainset_df.head(i), reader)
        partial_trainset = partial_trainset.build_full_trainset()
        model.fit(partial_trainset)
        predictions = model.test(testset)

        new_line = {
            "nr_items": i,
        }

        if comparison_method == "prf":
            precision_score, recall_score, f1_score = eval_func.precision_recall_f1(
                testset, predictions, k_value
            )

            new_line["precision_score"] = precision_score
            new_line["recall_score"] = recall_score
            new_line["f1_score"] = f1_score

        elif comparison_method == "mae_rmse":
            new_line["mae_score"] = mae(predictions, verbose=False)
            new_line["rmse_score"] = rmse(predictions, verbose=False)

        table.append(new_line)

    return jsonify(table)


if __name__ == "__main__":
    app.run(port=8000, debug=True, threaded=True)
