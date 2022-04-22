from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m = len(y)
        self.classes_, class_indexes, counts = np.unique(y, return_inverse=True, return_counts=True)
        self.pi_ = counts/m

        self.mu_ = []
        for k in range(len(self.classes_)):
            vec = np.zeros(X.shape[1])
            for row in range(len(X)):
                if y[row] == self.classes_[k]:
                    vec += X[row]
            vec = vec / counts[k]
            self.mu_.append(vec)
        self.mu_ = np.array(self.mu_)

        self.cov_ = np.zeros((X.shape[1],X.shape[1]))
        for i in range(len(X)):
            centered = np.atleast_2d(X[i] - self.mu_[class_indexes[i]])
            mult = centered.T @ centered
            self.cov_ += mult
        self.cov_ /= (m-len(self.classes_))

        self._cov_inv = np.linalg.inv(self.cov_)
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        # def get_y(x: np.ndarray):
        #     fs = []
        #     for i in range(len(self.classes_)):
        #         a = self._cov_inv @ self.mu_[i]
        #         b = np.log(self.pi_[i]) - 0.5 * (self.mu_[i] @ self._cov_inv @ self.mu_[i].T)
        #         fs.append(a @ x.T + b)
        #     return self.classes_[np.argmax(fs)]
        #
        # return np.apply_along_axis(get_y, axis=0)

        likelihoods = self.likelihood(X)
        pred = []
        for l in likelihoods:
            pred.append(self.classes_[np.argmax(l)])

        return np.array(pred)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        in_sqrt = np.power(2*np.pi, X.shape[1]) * np.linalg.det(self.cov_)
        z = 1/np.sqrt(in_sqrt)

        likelihoods = []
        for i in range(len(X)):
            i_likelihood = []
            for k in range(len(self.classes_)):
                mu = self.mu_[k]
                pi = self.pi_[k]
                centered_x = X[i] - mu
                in_exp = -0.5 * centered_x @ self._cov_inv @ centered_x.T

                i_likelihood.append(z * np.exp(in_exp) * pi)
            likelihoods.append(np.array(i_likelihood))

        return np.array(likelihoods)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
