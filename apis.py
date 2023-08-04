from flask import jsonify, Flask
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae

import time
import utils
import eval_functions as eval_func
import pandas as pd

app = Flask(__name__)

@app.route("/api/v1/mae_rmse/<model_name>", methods=["GET"])
def get_mae_rmse_model(model_name):
    data = Dataset.load_builtin("ml-100k")

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