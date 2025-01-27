from sklearn.model_selection import ParameterGrid

import IMLearn.metrics
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from scipy import stats

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    df = pd.read_csv(filename).drop_duplicates()

    df["floors"] = df["floors"].astype(float)
    df["bathrooms"] = df["bathrooms"].astype(int)
    df["bedrooms"] = df["bedrooms"].astype(float)

    # split date column to three int columns with year, month, date
    splitted_ymd = df["date"].str.extract(r'([\d]{4})([\d]{2})([\d]{2})', expand=True)
    df[["buy_year", "buy_month", "buy_day"]] = splitted_ymd[[0, 1, 2]].fillna(1).astype(int)

    # start by removing samples with impossible features
    df = df[df["bedrooms"] > 0]
    df = df[df["price"] > 0]
    df = df[df["floors"] > 0]

    df = df[df["waterfront"] >= 0]
    df = df[df["waterfront"] <= 1]
    df = df[df["bathrooms"] > 0]

    df = df[df["sqft_living"] > 0]
    df = df[df["sqft_living15"] > 0]
    df = df[df["sqft_lot"] > 0]
    df = df[df["sqft_lot15"] > 0]

    df = df[df["yr_built"] > 0]
    df = df[df["yr_built"] <= 2022]
    df = df[df["yr_renovated"] >= 0]
    df = df[df["yr_renovated"] <= 2022]

    df = df[df["buy_year"] > 0]
    df = df[df["buy_year"] <= 2022]

    df = df[df["buy_month"] > 0]
    df = df[df["buy_month"] < 13]

    df = df[df["buy_day"] > 0]
    df = df[df["buy_day"] < 32]

    df = pd.concat([df, pd.get_dummies(df["view"], prefix="View Type", prefix_sep=": ")], axis=1)
    df["Renovated Last Decade"] = (df["yr_renovated"] > 2010).astype(int)

    labels = df["price"]
    filtered_fits = ["price", "date", "id", "zipcode", "yr_renovated", "view", "buy_day", "buy_month"]
    fits = [f for f in df.columns if f not in filtered_fits]

    features = df[fits]
    feature_evaluation(features, labels)
    return features, labels


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = "../pearson_correletions.png") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    def pearson_correlation(x: np.ndarray,y: np.ndarray):
        x_centered = x - x.mean()
        y_centered = y - y.mean()

        mone = np.inner(x_centered,y_centered)
        mechane = np.linalg.norm(x_centered)*np.linalg.norm(y_centered)

        return mone/mechane

    np_df = X.to_numpy()
    np_y = y.to_numpy()

    correlations = []
    for i in range(len(np_df[0])):
        correlations.append(pearson_correlation(np_df[:,i], np_y))

    min_feat = str(X.columns[np.argmin(correlations)])
    max_feat = str(X.columns[np.argmax(correlations)])

    min_fig = scatter_feature(X[min_feat], np_y, min_feat, "Price")
    max_fig = scatter_feature(X[max_feat], np_y, max_feat, "Price")
    min_fig.show()
    max_fig.show()
    min_fig.write_image("../Correlations/min feature.png")
    max_fig.write_image("../Correlations/max feature.png")


def scatter_feature(feature: np.ndarray, label: np.ndarray, feature_name: str, label_name: str):
    fig = go.Figure([go.Scatter(x=feature, y=label, mode='markers', name="Correlations")],
                    layout=go.Layout(title="r$\\text{Geometric representation of the correlation between " + feature_name + " and " + label_name + "}$",
                                     xaxis_title="$\\text{"+feature_name+"}$",
                                     yaxis_title="$\\text{"+label_name+"}$",
                                     height=500))
    return fig


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    features, labels = load_data("..\datasets\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    scatter_feature(features["long"], labels, "Longitude", "Price")
    scatter_feature(features["sqft_living"], labels, "Living Room Square-feet", "Price")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(features, labels, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    percentages = np.linspace(10,100,91)
    avg_loss = []
    loss_var = []
    loss_std = []
    for p in percentages:
        per_p = []
        for i in range(10):
            t_X, t_y, m, n = split_train_test(train_X, train_y, p / float(100))
            model = LinearRegression(False)
            model.fit(t_X.to_numpy(), t_y.to_numpy())
            per_p.append(model.loss(test_X.to_numpy(), test_y.to_numpy()))

        per_p = np.array(per_p)
        avg_loss.append(per_p.mean())
        loss_var.append(np.sqrt(per_p.var()))
        loss_std.append(per_p.std())

    avg_loss = np.array(avg_loss)
    loss_var = np.array(loss_var)
    loss_std = np.array(loss_std)


    fig = go.Figure([go.Scatter(x=percentages, y=avg_loss, mode='lines+markers', name="Average Loss"),
                      go.Scatter(x=percentages, y=avg_loss-2*loss_var, name="Avg Loss - 2 std(los)", fill="tonexty", mode="lines", line=dict(color="lightgrey"), showlegend=False),
                      go.Scatter(x=percentages, y=avg_loss+2*loss_var, name="Avg Loss + 2 std(los)",fill="tonexty", mode="lines", line=dict(color="lightgrey"), showlegend=False)],
                    layout=go.Layout(
                        title="r$\\text{Mean loss var by percentage of training sample with confidence intervals}$",
                        xaxis_title="$\\text{Percentage of training sample}$",
                        yaxis_title="$\\text{Average Loss}$"))
    fig.show()


