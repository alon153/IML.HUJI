from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m = len(y)
        self.classes_, class_indexes, counts = np.unique(y, return_inverse=True, return_counts=True)
        self.pi_ = counts / m

        self.mu_ = []
        for k in range(len(self.classes_)):
            vec = np.zeros(X.shape[1])
            for row in range(len(X)):
                if y[row] == self.classes_[k]:
                    vec += X[row]
            vec = vec / counts[k]
            self.mu_.append(vec)
        self.mu_ = np.array(self.mu_)

        sigs = np.zeros((len(self.classes_),X.shape[1]))
        for i in range(len(X)):
            for j in range(len(self.classes_)):
                if y[i] == self.classes_[j]:
                    square = (X[i] - self.mu_[j]) ** 2
                    sigs[j] += square / counts[j]
                    break

        self.vars_ = np.array(sigs)

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

        likelihoods = []
        for i in range(X.shape[0]):
            i_likelihood = []
            for k in range(len(self.classes_)):
                in_sqrt = 2 * np.pi * self.vars_[k][i]
                z = 1 / np.sqrt(in_sqrt)

                centerd = (X[i] - self.mu_[k])
                in_exp = -0.5 * (np.linalg.norm(centerd) ** 2) / self.vars_[k][i]

                i_likelihood.append(z * np.exp(in_exp) * self.pi_[k])
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
