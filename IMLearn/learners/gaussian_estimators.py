from __future__ import annotations
import numpy as np
import utils
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator
        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = X.mean()

        size = X.size if self.biased_ else (X.size - 1)
        sum_of_powers = np.sum(np.power(X - self.mu_, 2))
        self.var_ = sum_of_powers / size

        self.fitted_ = True
        return self

    # @staticmethod
    # def generic_pdf(mu: float, sigma: float, x: float) -> float:
    #     """
    #     Calculates PDF of Gaussian model with given expectation and variance
    #
    #     Parameters
    #     ----------
    #     mu : float
    #         Expectation of Gaussian
    #     sigma : float
    #         Variance of Gaussian
    #     X : float
    #         Sample to calculate PDF
    #
    #     Returns
    #     -------
    #     pdf: float
    #         PDF calculated
    #     """
    #     mechane = np.sqrt(2 * np.pi * sigma)
    #     mone = np.exp(-0.5 * (np.power(x - mu, 2) / sigma))
    #     return mone / mechane

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        # lpdf = lambda x: UnivariateGaussian.generic_pdf(self.mu_, self.var_, x)
        # return np.array(list(map(lpdf, X)))

        mechane = np.sqrt(2 * np.pi * self.var_)
        mone = np.exp(-0.5 * (np.square(X - self.mu_) / self.var_))
        return mone/mechane

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        n = len(X)
        x_centered = X - mu
        return -0.5 * n * np.log(2 * np.pi) - 0.5 * n * np.log(sigma) - np.inner(x_centered) / (2 * sigma)


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """

        self.mu_ = np.array(list(map(np.mean, X.transpose())))

        m = X.shape[0]
        centered = X - self.mu_
        self.cov_ = np.matmul(centered.transpose(), centered) / (m - 1)

        self.fitted_ = True
        return self

    # @staticmethod
    # def generic_pdf(mu: np.ndarray, cov: np.ndarray, x: np.ndarray) -> float:
    #     mechane = np.sqrt(np.power(2 * np.pi, x.size) * np.linalg.det(cov))
    #
    #     centered_x = x - mu
    #     mat_multiplication = np.linalg.multi_dot(centered_x, np.invert(cov), centered_x.transpose())
    #     mone = np.exp(-0.5 * mat_multiplication)
    #
    #     return mone / mechane


    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        d = len(X[0])
        x_centered = X - self.mu_
        cov_inv = np.linalg.inv(self.cov_)
        mechane = np.sqrt(np.power(2 * np.pi, d) * det(self.cov_))

        def single_mult(x):
            mul = x @ cov_inv * x.T
            return mechane * np.exp(-0.5 * mul)

        return np.apply_along_axis(single_mult, 1, x_centered)

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """

        m = len(X)
        d = len(X[0])
        cov_invert = inv(cov)
        x_centered = X - mu

        # the formula we got from Q9 in the theoretical part
        return -d * m / 2 * np.log(2 * np.pi) - m / 2 * np.log(det(cov)) \
               - 0.5 * np.sum(x_centered @ cov_invert * x_centered)
