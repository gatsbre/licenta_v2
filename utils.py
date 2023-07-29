import requests
import plotly.graph_objects as go
from surprise import SVD, KNNBasic, BaselineOnly, SlopeOne, CoClustering

def get_model_instance(model_name):
    model_factories = {
        "SVD": lambda: SVD(),
        "KNN": lambda: KNNBasic(),
        "Baseline": lambda: BaselineOnly(),
        "SlopeOne": lambda: SlopeOne(),
        "CoClustering": lambda: CoClustering(),
    }

    return model_factories[model_name]()

def get_mae_rmse_score_bars(model_name):
    bar_width = 0.1
    rmse_score, mae_score, elapsed_time = requests.get(f"http://127.0.0.1:8000/api/v1/mae_rmse/{model_name}").json()

    mae_bar = go.Bar(name=f'{model_name}', x=['MAE'], y=[mae_score], width=bar_width, text=[f'{model_name} <br> MAE: {round(mae_score, 6)}'], textposition='inside', insidetextanchor='middle', hoverinfo='text', hovertext=f'Model: {model_name} <br>MAE: {mae_score}')
    rmse_bar = go.Bar(name=f'{model_name}', x=['RMSE'], y=[rmse_score], width=bar_width, text=[f'{model_name} <br> RMSE: {round(rmse_score, 6)}'], textposition='inside', insidetextanchor='middle', hoverinfo='text', hovertext=f'Model: {model_name} <br>RMSE: {rmse_score}')
    time_bar = go.Bar(name=f'{model_name}', x=['Time'], y=[elapsed_time], width=bar_width, text=[f'{model_name} <br> Time: {round(elapsed_time, 6)}s'], textposition='outside', hoverinfo='text', hovertext=f'Model: {model_name} <br>Time: {round(elapsed_time, 6)}s')
    
    return mae_bar, rmse_bar, time_bar

def get_plots(selected_models):
    plots = []
    for model_name in selected_models:
        for i, score_bar in enumerate(get_mae_rmse_score_bars(model_name)):
            try:
                if not plots[i]:
                    plots[i] = []
            except IndexError:
                plots.append([])
            plots[i].append(score_bar)

    return plots