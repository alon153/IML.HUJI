from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    var = 1
    m = 1000

    X = np.random.normal(mu, var, m)
    uni_gaus = UnivariateGaussian().fit(X)
    print((uni_gaus.mu_, uni_gaus.var_))

    # Question 2 - Empirically showing sample mean is consistent
    start = 10
    jumps = 10

    mus = []
    ms = list(range(start, m + 1, jumps))
    for m in ms:
        uni_gaus.fit(X[:m])
        mus.append(uni_gaus.mu_)

    mus = np.array(mus)
    fig = go.Figure([go.Scatter(x=ms, y=np.abs(mus - mu), mode='lines', name="1"),
                     go.Scatter(x=ms, y=np.zeros(len(ms)), mode='lines', name="2")],
                    layout=go.Layout(title="$\\text{Error of Estimated Expectation}$",
                                     xaxis_title="$m\\text{ - number of samples}$",
                                     yaxis_title="r$|\hat\mu - \mu|$",
                                     height=600))
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    X_sorted = np.sort(X)
    fig = go.Figure([go.Scatter(x=X_sorted, y=uni_gaus.pdf(X_sorted), mode='lines', name="1")],
                    layout=go.Layout(title="$\\text{Error of Estimated Expectation}$",
                                     xaxis_title="$m\\text{ - number of samples}$",
                                     yaxis_title="r$|\hat\mu - \mu|$",
                                     height=600))
    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    m = 1000

    X = np.random.multivariate_normal(mu, cov, m)
    multi_gaus = MultivariateGaussian()
    multi_gaus.fit(X)
    print(multi_gaus.mu_)
    print(multi_gaus.cov_)

    # Question 5 - Likelihood evaluation


    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
