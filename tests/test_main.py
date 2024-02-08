from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.main import encode_sex, get_predictions


def test_encode_sex():
    # Given
    input = pd.DataFrame({
        "Sex": ["male", "female"],
    })
    expected = pd.DataFrame({
        "Sex": [0, 1],
    })
    expected_not_mutable_input = input.copy()

    # When
    actual = encode_sex(input)

    # Then
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(input, expected_not_mutable_input)


def test_get_predictions():
    # Given
    input_df = pd.DataFrame({
        "PassengerId": [1, 2, 3],
        "Age": [22, 27, 32],
        "Fare": [7.25, 13.85, 69.55],
        "Sex": [0, 1, 0],
        "Pclass": [3, 2, 1],
    })
    input_model = DecisionTreeClassifier()
    input_model.predict = MagicMock(return_value=1)

    expected = pd.DataFrame({
        "PassengerId": [1, 2, 3],
        "Survived": [1, 1, 1],
    })
    expected_not_mutable_input = input_df.copy()

    # When
    actual = get_predictions(input_df, input_model)

    # Then
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(input_df, expected_not_mutable_input)
