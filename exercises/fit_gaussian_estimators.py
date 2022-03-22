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

    X = np.random.normal(mu, var, size=m)
    uni_gaus = UnivariateGaussian().fit(X)
    print("the mu and var of the uni-variate gaussian are:\n", (uni_gaus.mu_, uni_gaus.var_))
    print()

    # Question 2 - Empirically showing sample mean is consistent
    mus = []
    ms = np.linspace(10, 1000, 100).astype(int)
    for m in ms:
        uni_gaus.fit(X[:m])
        mus.append(uni_gaus.mu_)

    mus = np.array(mus)
    fig = go.Figure([go.Scatter(x=ms, y=np.abs(mus - mu), mode='lines', name="r$|\hat\mu - \mu|$"),
                     go.Scatter(x=ms, y=np.zeros(len(ms)), mode='lines', name="0")],
                    layout=go.Layout(title="$\\text{Error of Estimated Expectation}$",
                                     xaxis_title="$\\text{number of samples}$",
                                     yaxis_title="r$|\hat\mu - \mu|$",
                                     height=500)) \
        .update_yaxes(range=[0, 1])
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    X_sorted = np.sort(X)
    fig = go.Figure([go.Scatter(x=X_sorted, y=uni_gaus.pdf(X_sorted), mode='markers', name="PDF")],
                    layout=go.Layout(title="r$\\text{PDF of the uni-variate Gaussian distribution N(10,1)}$",
                                     xaxis_title="$\\text{sample values}$",
                                     yaxis_title="$\\text{PDF of sample}$",
                                     height=500))
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
    print("the estimated mu is:\n",multi_gaus.mu_)
    print()
    print("the estimated covariance matrix is:\n",multi_gaus.cov_)
    print()

    # Question 5 - Likelihood evaluation
    space = np.linspace(-10, 10, 200)
    heat = np.zeros((space.size, space.size))
    for i, f1 in enumerate(space):
        for j, f3 in enumerate(space):
            heat[i, j] = multi_gaus.log_likelihood(np.array([f1, 0, f3, 0]), cov, X)

    fig = go.Figure(go.Heatmap(x=space, y=space, z=heat),
                    layout=go.Layout(title="$\\text{Log-likelihood of multi-variate Gaussian distribution with }\mu = "
                                           "[0,y,0,x]$")) \
        .update_xaxes(title="fourth feature") \
        .update_yaxes(title="second feature")
    fig.show()

    # Question 6 - Maximum likelihood
    indices = np.unravel_index(heat.argmax(), heat.shape)

    max_log_likelihood = np.round(heat[indices[0], indices[1]], 3)
    max_f1 = np.round(space[indices[0]], 3)
    max_f3 = np.round(space[indices[1]], 3)

    print("The (f1,f3) values with the max log-likelihood are:\n", (max_f1,max_f3), "\nwith log-likelihood:\n",
          max_log_likelihood)


# if __name__ == '__main__':
#     np.random.seed(0)
#     test_univariate_gaussian()
#     test_multivariate_gaussian()


if __name__ == '__main__':
    X = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
          -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    uni_gauss = UnivariateGaussian().fit(X)
    print(uni_gauss.log_likelihood(1,1,X))
    print(uni_gauss.log_likelihood(10,1,X))