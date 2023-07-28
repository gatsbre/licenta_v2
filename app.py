from flask import Flask, render_template, request, jsonify
from surprise import BaselineOnly, Dataset, Reader, SVD, SVDpp, KNNBasic, NMF, SlopeOne, CoClustering
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae
import plotly.graph_objects as go
import requests

app = Flask(__name__)

# Load the dataset
data = Dataset.load_builtin('ml-100k')
reader = Reader(rating_scale=(1, 5))

train_data, test_data = train_test_split(data, train_size=0.8)

def get_model_instance(model_name):
    if model_name == 'svd':
        return SVD()
    elif model_name == 'knn':
        return KNNBasic()
    elif model_name == 'baseline':
        return BaselineOnly()
    elif model_name == 'svdpp':
        return SVDpp()
    elif model_name == 'nmf':
        return NMF()
    elif model_name == "slope_one":
        return SlopeOne()
    elif model_name == "co_clustering":
        return CoClustering()

@app.route('/')
def show_error_graph():
    bar_width = 0.1
    selected_models = request.args.getlist('model')

    if selected_models == "":
        selected_models = ['svd', 'knn', 'baseline', 'slope_one', 'co_clustering']

    mae_to_plot = []
    rmse_to_plot = []

    for model_name in selected_models:
        model_response = requests.get("http://127.0.0.1:8000/mae_rmse/" + model_name)
        rmse_score, mae_score = model_response.json()

        mae_to_plot.append(go.Bar(name=f'{model_name.upper()}', x=['MAE'], y=[mae_score], width=bar_width))
        rmse_to_plot.append(go.Bar(name=f'{model_name.upper()}', x=['RMSE'], y=[rmse_score], width=bar_width))

    mae_to_plot.sort(key=lambda bar: bar.y)
    rmse_to_plot.sort(key=lambda bar: bar.y)

    fig_mae = go.Figure(data=mae_to_plot)

    fig_mae.update_layout(title="MAEs",
                      barmode='group',
                      yaxis=dict(title="MAE"))
    
    fig_rmse = go.Figure(data=rmse_to_plot)

    fig_rmse.update_layout(title="RMSEs",
                      barmode='group',
                      yaxis=dict(title="RMSE"))
    
    return render_template('plot.html', plot=[fig_mae.to_html(), fig_rmse.to_html()])

@app.route('/mae_rmse/<model_name>', methods=['GET'])
def get_rmse_mae_model(model_name):
    model = get_model_instance(model_name)
    model.fit(train_data)
    predictions = model.test(test_data)
    rmse_score = rmse(predictions, verbose=False)
    mae_score = mae(predictions, verbose=False)

    return jsonify(rmse_score, mae_score)

if __name__ == '__main__':
    app.run(port=8000, debug=True)
