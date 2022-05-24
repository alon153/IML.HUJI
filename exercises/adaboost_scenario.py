import numpy as np
from typing import Tuple
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metalearners import AdaBoost
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adabeast = AdaBoost(DecisionStump, n_learners)
    adabeast.fit(train_X, train_y)

    ts = np.linspace(1, n_learners, n_learners).astype(int)
    train_err = np.array([adabeast.partial_loss(train_X, train_y, t) for t in ts])
    test_err = np.array([adabeast.partial_loss(test_X, test_y, t) for t in ts])

    fig = go.Figure([go.Scatter(name="Train error", x=ts, y=train_err),
                     go.Scatter(name="Test error", x=ts, y=test_err)])
    fig.update_layout(title="Training and test errors as a function of fitted learners. noise = "+str(noise))
    fig.update_xaxes(title="Number of fitted learners")
    fig.update_yaxes(title="Error")
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    all_X = np.concatenate((train_X, test_X), axis=0)
    all_Y = np.concatenate((train_y, test_y), axis=0)

    plot_shapes = np.zeros(all_Y.shape)
    plot_shapes[train_X.shape[0]:] = 1

    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{t} Iterations}}$" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        t_predict = lambda X: adabeast.partial_predict(X=X, T=t)
        fig.add_traces([decision_surface(t_predict, lims[0], lims[1], showscale=False),
                        go.Scatter(x=all_X[:, 0], y=all_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=all_Y, symbol=plot_shapes, colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(
        title=rf"$\textbf{{Decision bounds of AdaBoost based on different iteration counts. noise = {noise}}}$",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_i = ts[np.argmin(test_err)]
    acc = 1 - adabeast.partial_loss(test_X, test_y, best_i)
    ada_i = lambda X: adabeast.partial_predict(X, best_i)
    fig = go.Figure([decision_surface(ada_i, lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))])
    fig.update_layout(title=rf"$\textbf{{Decision bounds of AdaBoost with ensemble size = {best_i} and Accuracy = {acc}. noise = {noise}}}$")
    fig.show()

    # Question 4: Decision surface with weighted samples
    D = (adabeast.D_ / np.max(adabeast.D_)) * 5
    ada_i = lambda X: adabeast.partial_predict(X, T[-1])
    fig = go.Figure([decision_surface(ada_i, lims[0], lims[1], showscale=False),
                     go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                marker=dict(color=train_y, colorscale=[custom[0], custom[-1]],
                                            line=dict(color="black", width=1), size=D))])
    fig.update_layout(
        title=rf"$\textbf{{Decision bounds of Adaboost with {T[-1]} ensembles with size as function of weights. noise = {noise}}}$")
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
