from flask import Flask, render_template, jsonify
from plotly import graph_objects as go
import utils

app = Flask(__name__)

# Load the dataset
@app.route("/")
def show_main():
    return render_template("main.html")

@app.route("/mae_rmse")
def show_mae_rmse_graph():
    return render_template("mae_rmse.html")

@app.route("/robustness")
def show_robustness_graph():
    return render_template("robustness.html")

@app.route("/precision_recall_f1")
def show_precision_f1_recall():
    return render_template("precision_recall_f1.html")

@app.route("/mae_rmse/get_plots/<models>", methods=['GET'])
def get_mae_rmse_plots(models):
    selected_models = models.split(',') # = utils.get_models()

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

    return jsonify(
        {
            "mae_plot": fig_mae.to_plotly_json(),
            "rmse_plot": fig_rmse.to_plotly_json(),
            "time_plot": fig_time.to_plotly_json(),
        }
    )

@app.route("/precision_recall_f1/get_plots", methods=['GET'])
def get_precision_recall_f1_plots():
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

    return jsonify(
        {
            "precision_plot": fig_precision.to_plotly_json(),
            "recall_plot": fig_recall.to_plotly_json(),
            "f1_plot": fig_f1.to_plotly_json(),
        }
    )

@app.route("/robustness/get_plots", methods=['GET'])
def get_robustness_plots():
    # selected_models = utils.get_models()
    
    # k_value = utils.get_k_value()
    
    # comparison_name = utils.get_comparison_method()
    
    # if comparison_name
    
    # robustness_plot = utils.get_plots(
    #     selected_models, comparison_method, k_value
    # )
    
    # fig_robustness = go.Figure(data=robustness_plot)
    # fig_robustness.update_layout(title="Robustness", barmode='group')
    
    # return jsonify(
    #     {
    #         "robustness_plot": fig_robustness.to_plotly_json()
    #     }
    # )
    ...


if __name__ == "__main__":
    app.run(port=8080, debug=True)
