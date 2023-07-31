import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(train_data, test_data, to_continuous=False):
    scaler = StandardScaler()

    if to_continuous:
        train_data_processed = scaler.fit_transform(train_data)
        test_data_processed = scaler.transform(test_data)

        return train_data_processed[:, :-1], test_data_processed[:, :-1], \
            train_data_processed[:, -1].reshape((-1, 1)), test_data_processed[:, -1].reshape((-1, 1))
    else:
        # Use OneHotEncoder for the target values
        encoder = OneHotEncoder(sparse=False)

        train_data_processed = scaler.fit_transform(train_data[:, :-1])
        test_data_processed = scaler.transform(test_data[:, :-1])

        train_target = train_data[:, -1].reshape((-1, 1))
        train_target_encoded = encoder.fit_transform(train_target)

        test_target = test_data[:, -1].reshape((-1, 1))
        test_target_encoded = encoder.transform(test_target)

        return train_data_processed, test_data_processed, train_target_encoded, test_target_encoded