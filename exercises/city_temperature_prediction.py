import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """

    df = pd.read_csv(filename, parse_dates=["Date"])
    df = df.fillna(0)

    df["Temp"] = df["Temp"].astype(float)
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    df["Month"] = df["Month"].astype(int)

    df = df[df['Month'] > 0]
    df = df[df['Month'] < 13]

    df = df[df['Year'] >= 0]
    df = df[df['Year'] <= 2022]

    df = df[df['Day'] > 0]
    df = df[df['Day'] < 32]

    df = df[df["Temp"] >= -20]
    df = df[df["Temp"] <= 60]

    df['DayOfYear'] = df['Date'].dt.dayofyear

    df = pd.concat([df[[f for f in df.columns if f not in ["Date", "City"]]],
                    # pd.get_dummies(df["Country"]),
                    pd.get_dummies(df["City"])], axis=1)

    return df


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("..\datasets\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_df = df.loc[df["Country"] == "Israel", :]
    israel_df["Year"] = israel_df["Year"].astype(str)
    fig = px.scatter(israel_df, x="DayOfYear", y="Temp", color="Year",
                     title="r$\\text{Temperature by day of the year}$")
    fig.update_layout(xaxis_title="Day of year", yaxis_title="Temperature")
    fig.show()

    by_month = israel_df[["Month", "Temp"]]
    bars = by_month.groupby("Month").agg("std")
    bars["Month"] = pd.DataFrame(
        ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    bars_fig = px.bar(bars, x="Month", y="Temp", text_auto=True)
    bars_fig.update_layout(title="STD of temperature in Israel by month",
                           xaxis_title="Month",
                           yaxis_title="STD of Temp.")
    bars_fig.show()

    # Question 3 - Exploring differences between countries
    error_mean = df.groupby(["Country","Month"])["Temp"].agg(["mean","std"]).reset_index()
    error_lines = px.line(error_mean, x="Month", y="mean", error_y="std", line_group="Country", color="Country")
    error_lines.update_layout(title="Average Temperature by month with STD errors",
                              xaxis_title="Month",
                              yaxis_title="Temperature")
    error_lines.show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_Y, test_X, test_Y = split_train_test(israel_df["DayOfYear"], israel_df["Temp"])
    losses = []
    for k in range(10):
        model = PolynomialFitting(k+1)
        model.fit(train_X.to_numpy(), train_Y.to_numpy())
        losses.append([k+1,np.round(model.loss(test_X.to_numpy(),test_Y.to_numpy()), 2)])

    for l in losses:
        print("k =",l[0]," loss =",l[1])

    losses = np.array(losses)
    losses_df = pd.DataFrame(data=losses, columns=["k","Loss"])
    poly_bars = px.bar(losses_df, x="k",y="Loss", text_auto=True)
    poly_bars.update_layout(title="Mean of loss by degree of fitted polynom",
                            xaxis_title="k - degree of polynom",
                            yaxis_title="Mean of loss")

    poly_bars.show()

    # Question 5 - Evaluating fitted model on different countries
    model = PolynomialFitting(6)
    model.fit(train_X.to_numpy(), train_Y.to_numpy())
    errors_df = pd.DataFrame()
    errors_df["Country"] = pd.DataFrame([c for c in df["Country"].drop_duplicates().values if c != "Israel"])
    errors = []
    for country in errors_df["Country"]:
        country_df = df[df["Country"] == country]
        errors.append(np.round(model.loss(country_df["DayOfYear"],country_df["Temp"]), 2))

    errors_df["Loss"] = pd.Series(errors)

    loss_bars = px.bar(errors_df, x="Country", y="Loss", text_auto=True)
    loss_bars.show()


