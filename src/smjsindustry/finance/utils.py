# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""The SageMaker JumpStart Industry utils module."""

import json
import os
import re
from functools import lru_cache
from typing import Callable, Dict, Literal, Mapping

from pydantic import TypeAdapter, ValidationError
import pandas as pd
from smjsindustry.finance.constants import (
    IMAGE_CONFIG_FILE,
    ECR_URI_TEMPLATE,
    REPOSITORY,
    CONTAINER_IMAGE_VERSION,
)

FreqLiteral = Literal["D", "W", "M", "Q", "Y"]
FreqLabelHandler = Callable[[str], str]
ImageConfig = Mapping[str, str]

_IMAGE_CONFIG_VALIDATOR = TypeAdapter(dict[str, str])

__all__ = [
    "FreqLiteral",
    "get_freq_label",
    "load_image_uri_config",
    "retrieve_image",
    "FREQ_LABEL_MAP",
]


def _get_freq_label_by_day(date_value: str) -> str:
    """Gets frequency label for the date value which is aggregated by day.

    Args:
        date_value (str): The date value.

    Returns:
        str: The date value aggregated by day.
    """
    if not bool(re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", date_value)):
        raise ValueError("Date needs to be in yyyy-mm-dd format when freq is D")
    return date_value


def _get_freq_label_by_week(date_value: str) -> str:
    """Gets frequency label for the date value which is aggregated by week.

    Args:
        date_value (str): The date value.

    Returns:
        str: The date value aggregated by week.
    """
    if bool(re.match(r"^\d{4}W\d{1,2}$", date_value)):
        return date_value
    if not bool(re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", date_value)):
        raise ValueError("Date needs to be in yyyy-mm-dd format when freq is W")
    ts = pd.Timestamp(date_value)
    # Converted to f-string
    return f"{ts.year}W{ts.week}"


def _get_freq_label_by_month(date_value: str) -> str:
    """Gets frequency label for the date value which is aggregated by month.

    Args:
        date_value (str): The date value.

    Returns:
        str: The date value aggregated by month.
    """
    if bool(re.match(r"^\d{4}M\d{1,2}$", date_value)):
        return date_value
    if not bool(re.match(r"^\d{4}-\d{1,2}(-\d{1,2})?$", date_value)):
        raise ValueError("Date needs to be in yyyy-mm-dd or yyyy-mm format when freq is M")
    ts = pd.Timestamp(date_value)
    # Converted to f-string
    return f"{ts.year}M{ts.month}"


def _get_freq_label_by_quarter(date_value: str) -> str:
    """Gets frequency label for the date value which is aggregated by quarter.

    Args:
        date_value (str): The date value.

    Returns:
        str: The date value aggregated by quarter.
    """
    if bool(re.match(r"^\d{4}Q\d{1,2}$", date_value)):
        return date_value
    if not bool(re.match(r"^\d{4}-\d{1,2}(-\d{1,2})?$", date_value)):
        raise ValueError("Date needs to be in yyyy-mm-dd or yyyy-mm format when freq is Q")
    ts = pd.Timestamp(date_value)
    # Converted to f-string
    return f"{ts.year}Q{ts.quarter}"


def _get_freq_label_by_year(date_value: str) -> str:
    """Gets frequency label for the date value which is aggregated by year.

    Args:
        date_value (str): The date value.

    Returns:
        str: The date value aggregated by year.
    """
    if bool(re.match(r"^\d{4}$", date_value)):
        return date_value
    if not bool(re.match(r"^\d{4}(-\d{1,2}){0,2}$", date_value)):
        raise ValueError("Date needs to be in yyyy-mm-dd, yyyy-mm or yyyy format when freq is Y")
    ts = pd.Timestamp(date_value)
    return str(ts.year)


FREQ_LABEL_MAP: Dict[FreqLiteral, FreqLabelHandler] = {
    "D": _get_freq_label_by_day,
    "W": _get_freq_label_by_week,
    "M": _get_freq_label_by_month,
    "Q": _get_freq_label_by_quarter,
    "Y": _get_freq_label_by_year,
}


def get_freq_label(date_value: str, freq: str) -> str:
    """Gets frequency label for the date value.

    Args:
        date_value (str): The date value.
        freq (str): The frequency value specifies how the date field should be aggregated,
            by year, quarter, month, week, day. Available values:
            ``{'Y', 'Q', 'M', 'W', 'D'}``, default ``'Q'``.

    Returns:
        str: The date value aggregated by the specified frequency.
    """
    freq = freq.upper()
    handler = FREQ_LABEL_MAP.get(freq)
    if handler is None:
        raise ValueError(f"frequency {freq} not supported")
    if not isinstance(date_value, str):
        raise ValueError("The date column needs to be string")
    return handler(date_value.upper())


@lru_cache(maxsize=1)
def load_image_uri_config() -> ImageConfig:
    """Loads the JSON config for the image URI.

    Returns:
        JSON object: The JSON object of the image URI config.
    """
    fname = os.path.join(os.path.dirname(__file__), IMAGE_CONFIG_FILE)
    with open(fname) as f:
        config = json.load(f)

    try:
        return _IMAGE_CONFIG_VALIDATOR.validate_python(config)
    except ValidationError as exc:
        raise ValueError(f"Invalid image configuration in {fname}") from exc


def retrieve_image(region: str) -> str:
    """Retrieves the Amazon ECR image URI for the Docker image matching the given region.

    Args:
        region (str): The AWS Region.

    Returns:
        str: the Amazon ECR image URI for the corresponding Docker image.
    """
    config = load_image_uri_config()
    try:
        account_id = config[region]
    except KeyError as exc:
        raise ValueError(f"Region '{region}' is not supported in the image config.") from exc
    # NOTE: The CONTAINER_IMAGE_VERSION defined in constants.py must reference a
    # Python 3.11-compatible image tag for a successful processing job.
    # Converted to f-string
    repository = f"{REPOSITORY}:{CONTAINER_IMAGE_VERSION}"
    # Converted to f-string
    return ECR_URI_TEMPLATE.format(account_id=account_id, region=region, repository=repository)
