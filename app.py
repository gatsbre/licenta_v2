from flask import Flask, render_template, jsonify
from plotly import graph_objects as go
import utils
import math

app = Flask(
    __name__,
    static_url_path="",
    static_folder="web/static",
    template_folder="web/templates",
)


@app.route("/")
def show_main():
    return render_template("/pages/mae_rmse.html")


@app.route("/mae_rmse")
def show_mae_rmse_graph():
    return render_template("/pages/mae_rmse.html")


@app.route("/precision_recall_f1")
def show_precision_f1_recall():
    return render_template("/pages/precision_recall_f1.html")


@app.route("/robustness")
def show_robustness_graph():
    return render_template("/pages/robustness.html")


@app.route("/scarcity")
def show_scarcity_graph():
    return render_template("/pages/scarcity.html")


@app.route("/mae_rmse/get_plots/<models>/", methods=["GET"])
def get_mae_rmse_plots(models):
    selected_models = models.split(",")

    mae_to_plot, rmse_to_plot, time_to_plot = utils.get_plots(
        selected_models, utils.get_mae_rmse_score_bars
    )

    fig_mae = go.Figure(data=mae_to_plot)
    fig_mae.update_layout(
        title="Eroarea medie absoluta (MAE)",
        barmode="group",
        yaxis_range=[
            math.log10(min(mae_to_plot, key=lambda bar: bar["y"])["y"][0] * 0.98),
            math.log10(max(mae_to_plot, key=lambda bar: bar["y"])["y"][0] * 1.02),
        ],
    )
    fig_mae.update_yaxes(type="log")

    fig_rmse = go.Figure(data=rmse_to_plot)
    fig_rmse.update_layout(
        title="Radacina erorii mediei patratice (RMSE)",
        barmode="group",
        yaxis_range=[
            math.log10(min(rmse_to_plot, key=lambda bar: bar["y"])["y"][0] * 0.98),
            math.log10(max(rmse_to_plot, key=lambda bar: bar["y"])["y"][0] * 1.02),
        ],
    )
    fig_rmse.update_yaxes(type="log")

    fig_time = go.Figure(data=time_to_plot)
    fig_time.update_layout(
        title="Timpul de antrenare si de predictie a modelelor",
        barmode="group",
        yaxis_range=[
            math.log10(min(time_to_plot, key=lambda bar: bar["y"])["y"][0] * 0.98),
            math.log10(max(time_to_plot, key=lambda bar: bar["y"])["y"][0] * 1.6),
        ],
    )
    fig_time.update_yaxes(type="log")

    return jsonify(
        {
            "mae_plot": fig_mae.to_plotly_json(),
            "rmse_plot": fig_rmse.to_plotly_json(),
            "time_plot": fig_time.to_plotly_json(),
        }
    )


@app.route("/precision_recall_f1/get_plots/<int:k_value>/<models>/", methods=["GET"])
def get_precision_recall_f1_plots(models, k_value):
    selected_models = models.split(",")

    precision_to_plot, recall_to_plot, f1_to_plot = utils.get_plots(
        selected_models, utils.get_precision_recall_f1_score_bars, k_value
    )

    fig_precision = go.Figure(data=precision_to_plot)
    fig_precision.update_layout(
        title="Precision",
        barmode="group",
        yaxis_range=[
            math.log10(min(precision_to_plot, key=lambda bar: bar["y"])["y"][0] * 0.98),
            math.log10(max(precision_to_plot, key=lambda bar: bar["y"])["y"][0] * 1.02),
        ],
    )

    fig_precision.update_yaxes(
        type="log",
    )

    fig_recall = go.Figure(data=recall_to_plot)
    fig_recall.update_layout(
        title="Recall",
        barmode="group",
        yaxis_range=[
            math.log10(min(recall_to_plot, key=lambda bar: bar["y"])["y"][0] * 0.98),
            math.log10(max(recall_to_plot, key=lambda bar: bar["y"])["y"][0] * 1.02),
        ],
    )
    fig_recall.update_yaxes(type="log")

    fig_f1 = go.Figure(data=f1_to_plot)
    fig_f1.update_layout(
        title="F1",
        barmode="group",
        yaxis_range=[
            math.log10(min(f1_to_plot, key=lambda bar: bar["y"])["y"][0] * 0.98),
            math.log10(max(f1_to_plot, key=lambda bar: bar["y"])["y"][0] * 1.02),
        ],
    )
    fig_f1.update_yaxes(type="log")

    return jsonify(
        {
            "precision_plot": fig_precision.to_plotly_json(),
            "recall_plot": fig_recall.to_plotly_json(),
            "f1_plot": fig_f1.to_plotly_json(),
        }
    )


@app.route(
    "/robustness/get_plots/<int:nr_users>/<float:rating>/<comparison_method>/<k>/<models>",
    methods=["GET"],
)
def get_robustness_plots(models, nr_users, rating, comparison_method, k):
    selected_models = models.split(",")

    if comparison_method == "mae_rmse":
        mae_to_plot, rmse_to_plot = utils.get_plots(
            selected_models,
            utils.get_robustness_score_bars,
            nr_users=nr_users,
            rating=rating,
            comparison_method=comparison_method,
            k=k,
        )

        fig_robustness_mae = go.Figure(data=mae_to_plot)
        fig_robustness_mae.update_layout(
            title="Robustness MAE",
            barmode="group",
            yaxis_range=[
                math.log10(min(mae_to_plot, key=lambda bar: bar["y"])["y"][0] * 0.98),
                math.log10(max(mae_to_plot, key=lambda bar: bar["y"])["y"][0] * 1.02),
            ],
        )
        fig_robustness_mae.update_yaxes(type="log")

        fig_robustness_rmse = go.Figure(data=rmse_to_plot)
        fig_robustness_rmse.update_layout(
            title="Robustness RMSE",
            barmode="group",
            yaxis_range=[
                math.log10(min(rmse_to_plot, key=lambda bar: bar["y"])["y"][0] * 0.98),
                math.log10(max(rmse_to_plot, key=lambda bar: bar["y"])["y"][0] * 1.02),
            ],
        )
        fig_robustness_rmse.update_yaxes(type="log")

        return jsonify(
            {
                "robustness_plot_1": fig_robustness_mae.to_plotly_json(),
                "robustness_plot_2": fig_robustness_rmse.to_plotly_json(),
            }
        )
    elif comparison_method == "prf":
        precision_to_plot, recall_to_plot, f1_to_plot = utils.get_plots(
            selected_models,
            utils.get_robustness_score_bars,
            nr_users=nr_users,
            rating=rating,
            comparison_method=comparison_method,
            k=k,
        )

        fig_robustness_precision = go.Figure(data=precision_to_plot)
        fig_robustness_precision.update_layout(
            title="Robustness Precision",
            barmode="group",
            yaxis_range=[
                math.log10(
                    min(precision_to_plot, key=lambda bar: bar["y"])["y"][0] * 0.98
                ),
                math.log10(
                    max(precision_to_plot, key=lambda bar: bar["y"])["y"][0] * 1.02
                ),
            ],
        )
        fig_robustness_precision.update_yaxes(type="log")

        fig_robustness_recall = go.Figure(data=recall_to_plot)
        fig_robustness_recall.update_layout(
            title="Robustness Recall",
            barmode="group",
            yaxis_range=[
                math.log10(
                    min(recall_to_plot, key=lambda bar: bar["y"])["y"][0] * 0.98
                ),
                math.log10(
                    max(recall_to_plot, key=lambda bar: bar["y"])["y"][0] * 1.02
                ),
            ],
        )
        fig_robustness_recall.update_yaxes(type="log")

        fig_robustness_f1 = go.Figure(data=f1_to_plot)
        fig_robustness_f1.update_layout(
            title="Robustness F1",
            barmode="group",
            yaxis_range=[
                math.log10(min(f1_to_plot, key=lambda bar: bar["y"])["y"][0] * 0.98),
                math.log10(max(f1_to_plot, key=lambda bar: bar["y"])["y"][0] * 1.02),
            ],
        )
        fig_robustness_f1.update_yaxes(type="log")

        return jsonify(
            {
                "robustness_plot_1": fig_robustness_precision.to_plotly_json(),
                "robustness_plot_2": fig_robustness_recall.to_plotly_json(),
                "robustness_plot_3": fig_robustness_f1.to_plotly_json(),
            }
        )


@app.route(
    "/scarcity/get_plots/<models>/<model_feeding_rate>/<comparison_method>/<k>",
    methods=["GET"],
)
def get_scarcity_plots(models, model_feeding_rate, comparison_method, k=10):
    selected_models = models.split(",")

    if comparison_method == "mae_rmse":
        mae_to_plot, rmse_to_plot = utils.get_plots(
            selected_models=selected_models,
            function=utils.get_scarcity_mae_rmse,
            comparison_method=comparison_method,
            model_feeding_rate=model_feeding_rate,
        )

        fig_scarcity_mae = go.Figure(data=mae_to_plot)
        fig_scarcity_mae.update_layout(title="Scarcity MAE")
        fig_scarcity_mae.update_yaxes(type="log")

        fig_scarcity_rmse = go.Figure(data=rmse_to_plot)
        fig_scarcity_rmse.update_layout(title="Scarcity RMSE")
        fig_scarcity_rmse.update_yaxes(type="log")

        return jsonify(
            {
                "scarcity_plot_1": fig_scarcity_mae.to_plotly_json(),
                "scarcity_plot_2": fig_scarcity_rmse.to_plotly_json(),
            }
        )
    elif comparison_method == "prf":
        precision_to_plot, recall_to_plot, f1_to_plot = utils.get_plots(
            selected_models=selected_models,
            function=utils.get_scarcity_prf,
            comparison_method=comparison_method,
            model_feeding_rate=model_feeding_rate,
            k=k,
        )

        fig_scarcity_precision = go.Figure(data=precision_to_plot)
        fig_scarcity_precision.update_layout(
            title="Scarcity Precision",
        )
        fig_scarcity_precision.update_yaxes(type="log")

        fig_scarcity_recall = go.Figure(data=recall_to_plot)
        fig_scarcity_recall.update_layout(title="Scarcity Recall")
        fig_scarcity_recall.update_yaxes(type="log")

        fig_scarcity_f1 = go.Figure(data=f1_to_plot)
        fig_scarcity_f1.update_layout(title="Scarcity F1")
        fig_scarcity_f1.update_yaxes(type="log")

        return jsonify(
            {
                "scarcity_plot_1": fig_scarcity_precision.to_plotly_json(),
                "scarcity_plot_2": fig_scarcity_recall.to_plotly_json(),
                "scarcity_plot_3": fig_scarcity_f1.to_plotly_json(),
            }
        )


if __name__ == "__main__":
    app.run(port=8080, debug=True, threaded=True)
