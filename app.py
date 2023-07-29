from flask import Flask, render_template, request, jsonify
from surprise import  Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae
import plotly.graph_objects as go

import time
import utils

app = Flask(__name__)

# Load the dataset
data = Dataset.load_builtin('ml-100k')
reader = Reader(rating_scale=(1, 5))

train_data, test_data = train_test_split(data, train_size=0.8)

@app.route('/mae_rmse')
def show_mae_rmse_graph():
    selected_models = request.args.getlist('model') or ['SVD', 'KNN', 'Baseline', 'SlopeOne', 'CoClustering']
    mae_to_plot, rmse_to_plot, time_to_plot = utils.get_plots(selected_models)

    fig_mae = go.Figure(data=mae_to_plot)
    fig_mae.update_layout(title="Eroarea medie absoluta (MAE)", barmode='group')

    fig_rmse = go.Figure(data=rmse_to_plot)
    fig_rmse.update_layout(title="Radacina erorii mediei patratice (RMSE)", barmode='group')
    
    fig_time = go.Figure(data=time_to_plot)
    fig_time.update_layout(title="Timpul de antrenare si de predictie a modelelor", barmode='group', yaxis=dict(range=(0, max(time_to_plot, key=lambda time: time['y'])['y'][0] + 0.2)))

    return render_template('mae_rmse.html', mae_plot=fig_mae.to_html(), rmse_plot=fig_rmse.to_html(), time_plot=fig_time.to_html())

@app.route('/precision_f1_recall')
def show_precision_f1_recall():
    selected_models = request.args.getlist('model') or ['SVD', 'KNN', 'Baseline', 'SlopeOne', 'CoClustering']

@app.route('/')
def show_main():
    ...

@app.route('/api/v1/mae_rmse/<model_name>', methods=['GET'])
def get_mae_rmse_model(model_name):
    model = utils.get_model_instance(model_name)
    start_time = time.time()
    model.fit(train_data)

    predictions = model.test(test_data)
    end_time = time.time()
    mae_score = mae(predictions, verbose=False)
    rmse_score = rmse(predictions, verbose=False)

    return jsonify(rmse_score, mae_score, end_time-start_time)

if __name__ == '__main__':
    app.run(port=8000, debug=True)