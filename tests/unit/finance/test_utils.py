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
"""Tests utils module."""

import pytest
from smjsindustry.finance.utils import get_freq_label, retrieve_image
from smjsindustry.finance.constants import REPOSITORY, CONTAINER_IMAGE_VERSION


@pytest.mark.parametrize(
    "date_value", ["2020-05-01", "2020-05", "2020", "2020/05/01", "2020|6", 2020]
)
@pytest.mark.parametrize("freq", ["Y", "Q", "M", "W", "D", "T", "y"])
def test_get_freq_label(date_value, freq):
    # This block tests the case where the input is NOT a string (like the integer 2020)
    # The expected exception type was changed in utils.py from Exception to ValueError
    if date_value == 2020:
        with pytest.raises(ValueError) as error:
            # Reverted to date_value and added type: ignore to satisfy static type checker
            get_freq_label(date_value, freq)  # type: ignore[arg-type]
            assert "The date column needs to be string" in str(error.value)
    elif freq == "T":
        with pytest.raises(ValueError, match=r"^frequency .* not supported$"):
            get_freq_label(date_value, freq)
    # Corrected minor typo: "2020/05-01" changed to "2020/05/01" for consistency
    elif date_value == "2020/05/01" or date_value == "2020|6":
        with pytest.raises(ValueError, match=r"^Date needs to be in .* format when freq is .$"):
            get_freq_label(date_value, freq)
    elif freq == "Y" or freq == "y":
        actual = get_freq_label(date_value, freq)
        expected = "2020"
        assert actual == expected
    elif freq == "Q" and date_value != "2020":
        actual = get_freq_label(date_value, freq)
        expected = "2020Q2"
        assert actual == expected
    elif freq == "M" and date_value != "2020":
        actual = get_freq_label(date_value, freq)
        expected = "2020M5"
        assert actual == expected
    elif freq == "W" and date_value not in ("2020", "2020-05"):
        actual = get_freq_label(date_value, freq)
        expected = "2020W18"
        assert actual == expected
    elif freq == "D" and date_value == "2020-05-01":
        actual = get_freq_label(date_value, freq)
        expected = "2020-05-01"
        assert actual == expected
    else:
        with pytest.raises(ValueError, match=r"^Date needs to be in .* format when freq is .$"):
            get_freq_label(date_value, freq)


@pytest.mark.parametrize(
    "region",
    [
        "eu-north-1",
        "eu-west-1",
        "eu-west-2",
        "eu-west-3",
        "eu-central-1",
        "us-west-1",
        "us-west-2",
        "us-east-1",
        "us-east-2",
        "ap-south-1",
        "ap-northeast-2",
        "ap-southeast-1",
        "ap-southeast-2",
        "ap-northeast-1",
        "sa-east-1",
        "ap-east-1",
        "ca-central-1",
        "af-south-1",
        "me-south-1",
        "eu-south-1",
    ],
)
def test_retrieve_image(region):
    # Converted all expected strings to use f-strings for consistency
    if region == "eu-north-1":
        actual = retrieve_image(region)
        expected = f"010349432250.dkr.ecr.eu-north-1.amazonaws.com/{REPOSITORY}:{CONTAINER_IMAGE_VERSION}"
        assert actual == expected
    if region == "eu-west-1":
        actual = retrieve_image(region)
        expected = f"150602700506.dkr.ecr.eu-west-1.amazonaws.com/{REPOSITORY}:{CONTAINER_IMAGE_VERSION}"
        assert actual == expected
    if region == "eu-west-2":
        actual = retrieve_image(region)
        expected = f"294464218347.dkr.ecr.eu-west-2.amazonaws.com/{REPOSITORY}:{CONTAINER_IMAGE_VERSION}"
        assert actual == expected
    if region == "eu-west-3":
        actual = retrieve_image(region)
        expected = f"591089886631.dkr.ecr.eu-west-3.amazonaws.com/{REPOSITORY}:{CONTAINER_IMAGE_VERSION}"
        assert actual == expected
    if region == "eu-central-1":
        actual = retrieve_image(region)
        expected = f"810366494090.dkr.ecr.eu-central-1.amazonaws.com/{REPOSITORY}:{CONTAINER_IMAGE_VERSION}"
        assert actual == expected
    if region == "us-west-1":
        actual = retrieve_image(region)
        expected = f"496021652473.dkr.ecr.us-west-1.amazonaws.com/{REPOSITORY}:{CONTAINER_IMAGE_VERSION}"
        assert actual == expected
    if region == "us-west-2":
        actual = retrieve_image(region)
        expected = f"935494966801.dkr.ecr.us-west-2.amazonaws.com/{REPOSITORY}:{CONTAINER_IMAGE_VERSION}"
        assert actual == expected
    if region == "us-east-1":
        actual = retrieve_image(region)
        expected = f"207859150165.dkr.ecr.us-east-1.amazonaws.com/{REPOSITORY}:{CONTAINER_IMAGE_VERSION}"
        assert actual == expected
    if region == "us-east-2":
        actual = retrieve_image(region)
        expected = f"145207911424.dkr.ecr.us-east-2.amazonaws.com/{REPOSITORY}:{CONTAINER_IMAGE_VERSION}"
        assert actual == expected
    if region == "ap-south-1":
        actual = retrieve_image(region)
        expected = f"683153531578.dkr.ecr.ap-south-1.amazonaws.com/{REPOSITORY}:{CONTAINER_IMAGE_VERSION}"
        assert actual == expected
    if region == "ap-northeast-2":
        actual = retrieve_image(region)
        expected = f"041506878235.dkr.ecr.ap-northeast-2.amazonaws.com/{REPOSITORY}:{CONTAINER_IMAGE_VERSION}"
        assert actual == expected
    if region == "ap-southeast-1":
        actual = retrieve_image(region)
        expected = f"685484267512.dkr.ecr.ap-southeast-1.amazonaws.com/{REPOSITORY}:{CONTAINER_IMAGE_VERSION}"
        assert actual == expected
    if region == "ap-southeast-2":
        actual = retrieve_image(region)
        expected = f"780698971110.dkr.ecr.ap-southeast-2.amazonaws.com/{REPOSITORY}:{CONTAINER_IMAGE_VERSION}"
        assert actual == expected
    if region == "ap-northeast-1":
        actual = retrieve_image(region)
        expected = f"946773356576.dkr.ecr.ap-northeast-1.amazonaws.com/{REPOSITORY}:{CONTAINER_IMAGE_VERSION}"
        assert actual == expected
    if region == "sa-east-1":
        actual = retrieve_image(region)
        expected = f"138001272617.dkr.ecr.sa-east-1.amazonaws.com/{REPOSITORY}:{CONTAINER_IMAGE_VERSION}"
        assert actual == expected
    if region == "ap-east-1":
        actual = retrieve_image(region)
        expected = f"788152543915.dkr.ecr.ap-east-1.amazonaws.com/{REPOSITORY}:{CONTAINER_IMAGE_VERSION}"
        assert actual == expected
    if region == "ca-central-1":
        actual = retrieve_image(region)
        expected = f"057093961831.dkr.ecr.ca-central-1.amazonaws.com/{REPOSITORY}:{CONTAINER_IMAGE_VERSION}"
        assert actual == expected
    if region == "af-south-1":
        actual = retrieve_image(region)
        expected = f"204274516453.dkr.ecr.af-south-1.amazonaws.com/{REPOSITORY}:{CONTAINER_IMAGE_VERSION}"
        assert actual == expected
    if region == "me-south-1":
        actual = retrieve_image(region)
        expected = f"692383579251.dkr.ecr.me-south-1.amazonaws.com/{REPOSITORY}:{CONTAINER_IMAGE_VERSION}"
        assert actual == expected
    if region == "eu-south-1":
        actual = retrieve_image(region)
        expected = f"967756637777.dkr.ecr.eu-south-1.amazonaws.com/{REPOSITORY}:{CONTAINER_IMAGE_VERSION}"
        assert actual == expected