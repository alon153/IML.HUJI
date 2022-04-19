import pandas as pd

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    def create_callback(losses: list, X: np.ndarray, y: np.ndarray):
        def callback(p: Perceptron, sample: np.ndarray, classification: int):
            losses.append(p.loss(X,y))
        return callback

    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        title = n
        X, y = load_dataset("../datasets/"+f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        perceptron = Perceptron(callback=create_callback(losses, X, y))
        perceptron.fit(X,y)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure([go.Scatter(x=np.arange(1,len(losses)+1,1, dtype=int), y=np.array(losses))])
        fig.update_layout(title="Loss of Perceptron per iteration with "+title+" data")
        fig.update_layout(xaxis_title="Iterations")
        fig.update_layout(yaxis_title="Loss")
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X,y)

        bayes = GaussianNaiveBayes()
        bayes.fit(X,y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        fig = make_subplots(rows=1, cols=2)

        # lda_pred = lda.predict(X)
        # bayes_pred = bayes.predict(X)

        lda_pred = pd.DataFrame(lda.predict(X))
        bayes_pred = pd.DataFrame(bayes.predict(X))
        X = pd.DataFrame(X)
        X["true"] = pd.DataFrame(y).astype(str)
        X["bayes"] = bayes_pred.astype(str)
        X["lda"] = lda_pred.astype(str)

        print(lda_pred)
        print(bayes_pred)

        print(accuracy(y,lda_pred.to_numpy()))
        print(accuracy(y,bayes_pred.to_numpy()))

        px.scatter(X, x=X.columns[0], y=X.columns[1], color="lda", symbol="true").show()
        px.scatter(X, x=X.columns[0], y=X.columns[1], color="bayes", symbol="true").show()

        # fig.add_trace(px.scatter(X, x=X.columns[0], y=X.columns[1], color="lda", symbol="true"),row=1,col=1)
        # fig.add_trace(px.scatter(X, x=X.columns[0], y=X.columns[1], color="bayes", symbol="true"), row=1, col=1)
        # fig.show()


        # Add traces for data-points setting symbols and colors
        raise NotImplementedError()

        # Add `X` dots specifying fitted Gaussians' means
        raise NotImplementedError()

        # Add ellipses depicting the covariances of the fitted Gaussians
        raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
