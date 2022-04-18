import typing

from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from sklearn.model_selection import train_test_split
from typing import *
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from __future__ import annotations
from typing import NoReturn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC


def load_data(filename: str, is_training=True):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    full_data = pd.read_csv(filename)
    if is_training:
        full_data.drop_duplicates()

    # drop samples with booking date after checking date
    booking_dates = pd.to_datetime(full_data["booking_datetime"]).dt.date
    checkin_dates = pd.to_datetime(full_data["checkin_date"]).dt.date
    checkout_dates = pd.to_datetime(full_data["checkout_date"]).dt.date

    # add column for days of stay
    full_data["days_of_stay"] = (checkout_dates - checkin_dates).dt.days
    if is_training:
        full_data = full_data[booking_dates <= checkin_dates]
        # drop samples with checkout not-after checkin (one day)
        full_data = full_data[full_data["days_of_stay"] >= 1]

    # transform Nan in requests to 0
    special_requests = ["request_nonesmoke", "request_latecheckin", "request_highfloor",
                        "request_twinbeds", "request_airport"]
    for request in special_requests:
        full_data[request] = full_data[request].fillna(0)

    pay_int_replace = {0: -1, 1: 1}
    full_data = full_data.replace({"is_first_booking": pay_int_replace})

    full_data[special_requests] = np.where(full_data[special_requests] == 0, -1, 1)

    # standardize original selling amount
    #mean_selling_amount = full_data["original_selling_amount"].mean()

    # full_data["original_selling_amount"] /= mean_selling_amount
    full_data["original_selling_amount"] = np.log(full_data["original_selling_amount"])

    # full_data["price_per_day"] = \
    #     np.where(full_data["days_of_stay"] > 0, full_data["original_selling_amount"] / full_data["days_of_stay"], 0)

    full_data["4_and_up_rating"] = np.where(full_data["hotel_star_rating"] >= 4,1,-1)

    if is_training:
        # create labels - 1 for cancellation, 0 otherwise
        labels = full_data["cancellation_datetime"].between("2018-07-12", "2018-13-12").astype(int)
    else:
        labels = None

    numbers = ["no_of_room", "no_of_extra_bed", "no_of_children", "no_of_adults"]
    dummies = create_dummy_variables(full_data)

    features = full_data[
        ["days_of_stay", "4_and_up_rating", "original_selling_amount"] + special_requests + numbers]

    """
    Use this to handle cancel policies (for now raises loss...)
    """
    # handle_policies(full_data)
    # policy_features = ["cancel_in_policy", "policy_price"]
    # policy_features = ["policy_price"]
    # features = full_data[
    #     ["days_of_stay", "hotel_star_rating", "price_per_day"] + special_requests + numbers + policy_features]

    features = pd.concat([features, dummies], axis=1)

    return features, labels


def load_test_data(filename: str):
    """
    Load moodle test data
    """
    features = load_data(filename, False)[0]
    return features


def create_dummy_variables(df):
    return pd.concat([pd.get_dummies(df["charge_option"], drop_first=True),
                      pd.get_dummies(df["accommadation_type_name"], drop_first=True),
                      pd.get_dummies(df["guest_nationality_country_name"], drop_first=True),
                      pd.get_dummies(df["language"], drop_first=True),
                      pd.get_dummies(df["original_payment_currency"], drop_first=True),
                      pd.get_dummies(df["request_earlycheckin"], drop_first=True),
                      pd.get_dummies(df["hotel_chain_code"], drop_first=False),
                      pd.get_dummies(df["hotel_brand_code"], drop_first=False),
                      pd.get_dummies(df["original_payment_method"], drop_first=False)], axis=1)



def handle_policies(full_data):
    # full_data["has_no_show"] = np.where("_" in full_data["cancellation_policy_code"], 0, 1)
    split_policies = full_data["cancellation_policy_code"].str.split(r'_', expand=True)
    split_policies["checkin_date"] = full_data["checkin_date"]

    # split policy to days and price
    split_policies[["days_of_policy", "policy_price"]] = split_policies[0].str.extract(r'([\d]+)D([\w]+)')
    split_policies["days_of_policy"] = split_policies["days_of_policy"].fillna(0).astype(np.int64)

    # policy starts at "day_of_policy" before checkin
    split_policies["policy_start_date"] = pd.to_datetime(split_policies["checkin_date"]) - pd.to_timedelta(
        split_policies["days_of_policy"], unit='D')

    split_policies["days_of_stay"] = full_data["days_of_stay"]

    # split policy price to amount and type (e.g. (1,N) or (50,P))
    split_policies[["price_num", "price_type"]] = split_policies["policy_price"].str.extract(r'([\d]+)([\w])')
    split_policies["price_num"] = split_policies["price_num"].fillna(0).astype(float)

    # convert prices by night to prices by percentage (for unified price format)
    split_policies["price_num"] = np.where(split_policies["price_type"] == "N",
                                           split_policies["price_num"] / split_policies["days_of_stay"] * 100,
                                           split_policies["price_num"])
    split_policies["price_type"] = np.where(split_policies["price_type"] == "N",
                                            "P",
                                            "P")

    # check if 2018-07-12 is in the cancellation policy range
    start = "2018-07-12"
    before_start = np.where(
        split_policies["policy_start_date"].astype(str) < start, 1, -1)
    after_start = np.where(
        split_policies["checkin_date"].astype(str) > start, 1, -1)

    end = "2018-13-12"
    before_end = np.where(
        split_policies["policy_start_date"].astype(str) < end, 1, -1)
    after_end = np.where(
        split_policies["checkin_date"].astype(str) > end, 1, -1)

    split_policies["cancel_in_policy"] = np.where(before_start + after_start + before_end + after_end == 4, 1, -1)
    split_policies["cancel_in_policy"] = split_policies["cancel_in_policy"].astype(int)

    full_data[["cancel_in_policy", "policy_price"]] = split_policies[["cancel_in_policy", "price_num"]]

    # full_data["policy_price"] = split_policies["cancel_in_policy"] * split_policies["price_num"]
    full_data["policy_price"] = full_data["original_selling_amount"] * split_policies["price_num"]


def evaluate_and_export(estimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


def create_weights(X: pd.DataFrame, y: pd.Series, alpha: float = 0.03):
    def pearson_correlation(x: np.ndarray, y: np.ndarray):
        if x.mean() == 0:
            return 0
        x_centered = x - x.mean()
        y_centered = y - y.mean()

        mone = float(np.inner(x_centered, y_centered))
        mechane = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))

        return mone / mechane

    if alpha == 0:
        return

    np_df = X.to_numpy()
    np_y = y.to_numpy()

    # get correlation of each feature
    correlations = []
    for i in range(len(np_df[0])):
        correlations.append((pearson_correlation(np_df[:, i], np_y), X.columns[i]))

    correlations = sorted(correlations, key=lambda x: x[0])
    correlations = pd.DataFrame(correlations, columns=["correlations", "features"])
    correlations["features"] = correlations["features"].astype(str)
    correlations["correlations"] = correlations["correlations"].astype(float)

    # return Dataframe with a column of feature names and column of weights
    # weights are calculated by correlation/alpha. that way, the bigger the correlation is, the bigger the weight
    return pd.concat([correlations["features"], abs(correlations["correlations"] / alpha)], axis=1)


def apply_weights(X: pd.DataFrame, weights: pd.DataFrame):
    X_weighted = X.copy()
    for i in range(len(weights)):
        k = weights["features"][i]
        if k == "1.0":
            k = 1.0
        X_weighted[k] *= weights["correlations"][i]
    return X_weighted.to_numpy()

class AgodaCancellationEstimator():
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------
        """
        super().__init__()
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """
        #self.model = LogisticRegression().fit(X, y)
        #self.model = KNeighborsClassifier(5).fit(X,y)
        #self.model = SVC(max_iter = 1000)
        self.model = DecisionTreeClassifier().fit(X, y)

        #self.model.fit(X,y)
    def predict(self, X: np.ndarray) -> np.ndarray:
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

        #handle case where X's columns differ from the training columns

        return self.model.predict(X)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """
        return (self.predict(X) == y).sum() / len(X)

if __name__ == '__main__':
    np.random.seed(0)

    # Load train and test data, and align together
    df, cancellation_labels = load_data("../agoda_cancellation_train.csv", is_training=True)
    cancellation_labels = cancellation_labels.astype('int')

    test_df = load_test_data("../test_set_week_2.csv")

    df, test_df = df.align(test_df, axis=1, fill_value=0)

    X_train, X_test, y_train, y_test = train_test_split(df, cancellation_labels, test_size=0.2)

    """
    Use this to add weights (currently not working well)
    """
    #weights = create_weights(X_train, y_train)
    # X_train_w = apply_weights(X_train, weights)
    # X_test_w = apply_weights(X_test, weights)

    # Fit model over data
    estimator = AgodaCancellationEstimator()
    estimator.fit(X_train, y_train)

    # predict test
    pred = estimator.predict(test_df)
    print(estimator.loss(df, cancellation_labels))

    evaluate_and_export(estimator, test_df, "208242610_212359905_313163958.csv")
