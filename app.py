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
    model_factories = {
        "SVD": lambda: SVD(),
        "KNN": lambda: KNNBasic(),
        "Baseline": lambda: BaselineOnly(),
        "SlopeOne": lambda: SlopeOne(),
        "CoClustering": lambda: CoClustering(),
    }

    return model_factories[model_name]()

def get_score_bars(model_name):
    bar_width = 0.1
    rmse_score, mae_score = requests.get(f"http://127.0.0.1:8000/mae_rmse/{model_name}").json()

    mae_bar = go.Bar(name=f'{model_name}', x=['MAE'], y=[mae_score], width=bar_width, text=[f'{model_name}\n{round(mae_score, 4)}'], textposition='inside', insidetextanchor='middle', hoverinfo='text', hovertext=f'MAE: {mae_score}')
    rmse_bar = go.Bar(name=f'{model_name}', x=['RMSE'], y=[rmse_score], width=bar_width, text=[f'{model_name}\n{round(rmse_score, 4)}'], textposition='inside', insidetextanchor='middle', hoverinfo='text', hovertext=f'RMSE: {rmse_score}')
    
    return mae_bar, rmse_bar

def get_plots(selected_models):
    plots = []
    for model_name in selected_models:
        for i, score_bar in enumerate(get_score_bars(model_name)):
            try:
                if not plots[i]:
                    plots[i] = []
            except IndexError:
                plots.append([])
            plots[i].append(score_bar)

    return plots

@app.route('/')
def show_mae_rmse_graph():
    selected_models = request.args.getlist('model') or ['SVD', 'KNN', 'Baseline', 'SlopeOne', 'CoClustering']
    mae_to_plot, rmse_to_plot = get_plots(selected_models)

    fig_mae = go.Figure(data=mae_to_plot)
    fig_mae.update_layout(title="Eroarea medie absoluta (MAE)", barmode='group')

    fig_rmse = go.Figure(data=rmse_to_plot)
    fig_rmse.update_layout(title="Radacina erorii mediei patratice (RMSE)", barmode='group')

    return render_template('plot.html', mae_plot=fig_mae.to_html(), rmse_plot=fig_rmse.to_html())

@app.route('/mae_rmse/<model_name>', methods=['GET'])
def get_mae_rmse_model(model_name):
    model = get_model_instance(model_name)
    model.fit(train_data)

    predictions = model.test(test_data)

    mae_score = mae(predictions, verbose=False)
    rmse_score = rmse(predictions, verbose=False)

    return jsonify(rmse_score, mae_score)

if __name__ == '__main__':
    app.run(port=8000, debug=True)
