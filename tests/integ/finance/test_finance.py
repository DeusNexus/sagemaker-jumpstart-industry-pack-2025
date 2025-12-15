# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import io
import os
import shutil
import tempfile
from urllib.parse import urlparse

import boto3
import pandas as pd

from sagemaker.core.processing import Session

from smjsindustry import (
    NLPScoreType,
    NLPSCORE_NO_WORD_LIST,
    Summarizer,
    NLPScorer,
    NLPScorerConfig,
    JaccardSummarizerConfig,
    KMedoidsSummarizerConfig,
)
from smjsindustry.config import SEC_USER_AGENT_ENV, get_sec_user_agent
from smjsindustry.finance import DataLoader, SECXMLFilingParser, EDGARDataSetConfig
from tests.integ import DATA_DIR, timeout, utils

FINANCE_DEFAULT_TIMEOUT_MINUTES = 15
# Disable rich log streaming to avoid markup parsing errors in pytest output.
STREAM_PROCESSING_LOGS = False


def _resolve_sec_user_agent() -> str:
    """Return the SEC user agent supplied via configuration."""

    user_agent = get_sec_user_agent()
    if not user_agent:
        raise RuntimeError(
            "SEC user agent missing. Set the environment variable "
            f"{SEC_USER_AGENT_ENV} or DEFAULT_SEC_USER_AGENT before running the dataloader test."
        )
    return user_agent

# --- FIX 1: Helper function to manually construct the IAM Role ARN ---
def get_sagemaker_execution_role_arn(sagemaker_session, role_name="SageMakerRole"):
    """
    Constructs the full IAM Role ARN using the current account ID and the role name.
    This is necessary when 'iam:GetRole' permission is missing or the Session object
    lacks 'get_execution_role()'.
    """
    env_role_arn = os.getenv("SMJS_FINANCE_EXECUTION_ROLE_ARN")
    if env_role_arn:
        return env_role_arn
    # Use STS to get the Account ID of the current execution context
    sts = sagemaker_session.boto_session.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    
    # Manually construct the standard IAM Role ARN format: arn:aws:iam::ACCOUNT_ID:role/ROLE_NAME
    # NOTE: If this ARN is invalid, the role name "SageMakerRole" is incorrect for your account,
    # and should be replaced with the actual SageMaker execution role name.
    return f"arn:aws:iam::{account_id}:role/{role_name}"
# -------------------------------------------------------------------

def test_jaccard_summarizer(
    sagemaker_session,
    cpu_instance_type,
):
    jaccard_summarizer = None
    bucket = None
    prefix = None
    try:
        with timeout.timeout(minutes=FINANCE_DEFAULT_TIMEOUT_MINUTES):
            # --- FIX 2: Call the robust ARN helper function ---
            SM_EXECUTION_ROLE_ARN = get_sagemaker_execution_role_arn(sagemaker_session)
            # --------------------------------------------------

            jaccard_summarizer_config = JaccardSummarizerConfig(summary_size=100)
            data_path = os.path.join(DATA_DIR, "finance", "processor_data.csv")
            test_run = utils.unique_name_from_base("test_run")
            bucket = sagemaker_session.default_bucket()
            
            # UPDATED: f-string
            prefix = f"jumpstart-gecko-jaccard-summarizer/{test_run}"
            s3_output_path = f"s3://{bucket}/{prefix}"
            output_file_name = "output.csv"
            
            jaccard_summarizer = Summarizer(
                # Use the constructed ARN
                role=SM_EXECUTION_ROLE_ARN,
                instance_count=1,
                instance_type=cpu_instance_type,
                sagemaker_session=sagemaker_session,
            )
            jaccard_summarizer.summarize(
                jaccard_summarizer_config,
                "text",
                data_path,
                s3_output_path,
                output_file_name,
                new_summary_column_name="summary",
                logs=STREAM_PROCESSING_LOGS,
            )
        check_output_file_exists(jaccard_summarizer, sagemaker_session, bucket, prefix, output_file_name)
        check_output_file_new_columns(
            jaccard_summarizer, bucket, prefix, output_file_name, ["summary"]
        )
    finally:
        if jaccard_summarizer is not None and bucket is not None and prefix is not None:
            remove_test_resources(jaccard_summarizer, bucket, prefix)


def test_kmedoids_summarizer(
    sagemaker_session,
    cpu_instance_type,
):
    kmedoids_summarizer = None
    bucket = None
    prefix = None
    try:
        with timeout.timeout(minutes=FINANCE_DEFAULT_TIMEOUT_MINUTES):
            # --- FIX 2: Call the robust ARN helper function ---
            SM_EXECUTION_ROLE_ARN = get_sagemaker_execution_role_arn(sagemaker_session)
            # --------------------------------------------------

            kmedoids_summarizer_config = KMedoidsSummarizerConfig(100)
            data_path = os.path.join(DATA_DIR, "finance", "processor_data.csv")
            test_run = utils.unique_name_from_base("test_run")
            bucket = sagemaker_session.default_bucket()
            
            # UPDATED: f-string
            prefix = f"jumpstart-gecko-kmedoids-summarizer/{test_run}"
            s3_output_path = f"s3://{bucket}/{prefix}"
            output_file_name = "output.csv"
            
            kmedoids_summarizer = Summarizer(
                # Use the constructed ARN
                role=SM_EXECUTION_ROLE_ARN,
                instance_count=1,
                instance_type=cpu_instance_type,
                sagemaker_session=sagemaker_session,
            )
            kmedoids_summarizer.summarize(
                kmedoids_summarizer_config,
                "text",
                data_path,
                s3_output_path,
                output_file_name,
                new_summary_column_name="summary",
                logs=STREAM_PROCESSING_LOGS,
            )
        check_output_file_exists(kmedoids_summarizer, sagemaker_session, bucket, prefix, output_file_name)
        check_output_file_new_columns(
            kmedoids_summarizer, bucket, prefix, output_file_name, ["summary"]
        )
    finally:
        if kmedoids_summarizer is not None and bucket is not None and prefix is not None:
            remove_test_resources(kmedoids_summarizer, bucket, prefix)


def test_nlp_scorer(
    sagemaker_session,
    cpu_instance_type,
):
    nlp_scorer = None
    bucket = None
    prefix = None
    try:
        with timeout.timeout(minutes=FINANCE_DEFAULT_TIMEOUT_MINUTES):
            # --- FIX 2: Call the robust ARN helper function ---
            SM_EXECUTION_ROLE_ARN = get_sagemaker_execution_role_arn(sagemaker_session)
            # --------------------------------------------------
            
            score_type_list = list(
                NLPScoreType(score_type, [])
                for score_type in NLPScoreType.DEFAULT_SCORE_TYPES
                if score_type not in NLPSCORE_NO_WORD_LIST
            )
            score_type_list.extend(
                [NLPScoreType(score_type, None) for score_type in NLPSCORE_NO_WORD_LIST]
            )
            nlp_scorer_config = NLPScorerConfig(score_type_list)
            data_path = os.path.join(DATA_DIR, "finance", "processor_data.csv")
            test_run = utils.unique_name_from_base("test_run")
            bucket = sagemaker_session.default_bucket()
            
            # UPDATED: f-string
            prefix = f"jumpstart-gecko-nlp-scorer/{test_run}"
            s3_output_path = f"s3://{bucket}/{prefix}"
            output_file_name = "output.csv"
            
            nlp_scorer = NLPScorer(
                # Use the constructed ARN
                role=SM_EXECUTION_ROLE_ARN,
                instance_count=1,
                instance_type=cpu_instance_type,
                sagemaker_session=sagemaker_session,
            )
            nlp_scorer.calculate(
                nlp_scorer_config,
                "text",
                data_path,
                s3_output_path,
                output_file_name,
                logs=STREAM_PROCESSING_LOGS,
            )
        check_output_file_exists(nlp_scorer, sagemaker_session, bucket, prefix, output_file_name)
        check_output_file_new_columns(
            nlp_scorer, bucket, prefix, output_file_name, list(NLPScoreType.DEFAULT_SCORE_TYPES)
        )
    finally:
        if nlp_scorer is not None and bucket is not None and prefix is not None:
            remove_test_resources(nlp_scorer, bucket, prefix)


def test_dataloader(
    sagemaker_session,
    cpu_instance_type,
):
    dataloader = None
    bucket = None
    prefix = None
    local_output_dir = None
    try:
        with timeout.timeout(minutes=FINANCE_DEFAULT_TIMEOUT_MINUTES):
            # --- FIX 2: Call the robust ARN helper function ---
            SM_EXECUTION_ROLE_ARN = get_sagemaker_execution_role_arn(sagemaker_session)
            # --------------------------------------------------

            dataset_config = EDGARDataSetConfig(
                tickers_or_ciks=["amzn"],
                form_types=["10-Q"],
                filing_date_start="2020-01-01",
                filing_date_end="2020-03-31",
                email_as_user_agent=_resolve_sec_user_agent(),
            )
            local_fixture_path = os.getenv("SMJS_FINANCE_DATALOADER_LOCAL_DATASET")
            if local_fixture_path:
                local_output_dir = tempfile.mkdtemp(prefix="dataloader-output-")
                s3_output_path = f"file://{local_output_dir}"
            else:
                test_run = utils.unique_name_from_base("test_run")
                bucket = sagemaker_session.default_bucket()
                prefix = f"jumpstart-gecko-sec-filing-retrieval/{test_run}"
                s3_output_path = f"s3://{bucket}/{prefix}"
            output_file_name = "output.csv"
            
            dataloader = DataLoader(
                # Use the constructed ARN
                role=SM_EXECUTION_ROLE_ARN,
                instance_count=1,
                instance_type=cpu_instance_type,
                sagemaker_session=sagemaker_session,
            )
            dataloader.load(
                dataset_config,
                s3_output_path,
                output_file_name,
                logs=STREAM_PROCESSING_LOGS,
            )
        check_output_file_exists(dataloader, sagemaker_session, bucket, prefix, output_file_name)
        check_output_file_new_columns(
            dataloader,
            bucket,
            prefix,
            output_file_name,
            ["ticker", "form_type", "accession_number", "filing_date", "text"],
        )
    finally:
        if local_output_dir:
            shutil.rmtree(local_output_dir, ignore_errors=True)
        if dataloader is not None and bucket is not None and prefix is not None:
            remove_test_resources(dataloader, bucket, prefix)


def test_sec_xml_filing_parser(
    sagemaker_session,
    cpu_instance_type,
):
    parser = None
    bucket = None
    prefix = None
    try:
        with timeout.timeout(minutes=FINANCE_DEFAULT_TIMEOUT_MINUTES):
            # --- FIX 2: Call the robust ARN helper function ---
            SM_EXECUTION_ROLE_ARN = get_sagemaker_execution_role_arn(sagemaker_session)
            # --------------------------------------------------
            
            input_data_folder = os.path.join(DATA_DIR, "finance", "sec_filings")
            test_run = utils.unique_name_from_base("test_run")
            bucket = sagemaker_session.default_bucket()
            
            # UPDATED: f-string
            prefix = f"jumpstart-gecko-sec-parser/{test_run}"
            s3_output_path = f"s3://{bucket}/{prefix}"
            
            parser = SECXMLFilingParser(
                # Use the constructed ARN
                role=SM_EXECUTION_ROLE_ARN,
                instance_count=1,
                instance_type=cpu_instance_type,
                sagemaker_session=sagemaker_session,
            )
            parser.parse(
                input_data_folder,
                s3_output_path,
                logs=STREAM_PROCESSING_LOGS,
            )
        check_output_file_exists(parser, sagemaker_session, bucket, prefix, "parsed")
    finally:
        if parser is not None and bucket is not None and prefix is not None:
            remove_test_resources(parser, bucket, prefix)


def check_output_file_exists(processor, sagemaker_session, bucket, prefix, output_file_name):
    job_name = processor._current_job_name
    using_fixture = getattr(processor, "_local_fixture_used", False)
    fixture_uri = getattr(processor, "_local_fixture_output_uri", None)
    if not using_fixture:
        if not job_name:
            raise RuntimeError("Processor has not started a job; no output to validate.")
        if not processing_job_completed(sagemaker_session, job_name):
            raise AssertionError(f"Processing job {job_name} did not complete.")

    if using_fixture and fixture_uri:
        parsed = urlparse(fixture_uri)
        if parsed.scheme == "file":
            if not os.path.exists(parsed.path):
                raise AssertionError(f"Fixture output {parsed.path} missing.")
            return
        elif parsed.scheme == "s3":
            bucket = parsed.netloc
            prefix = os.path.dirname(parsed.path.lstrip("/"))

    region = processor.sagemaker_session.boto_region_name
    s3_client = boto3.client("s3", region_name=region)
    if bucket is None or prefix is None:
        raise AssertionError("Bucket/prefix required for S3 validation.")
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = response.get("Contents", [])
    if not contents:
        raise AssertionError(f"No objects found under prefix {prefix}")
    matching_keys = [obj["Key"] for obj in contents if output_file_name in obj["Key"]]
    assert matching_keys, f"{output_file_name} not found in {prefix}"


def check_output_file_new_columns(processor, bucket, prefix, output_file_name, new_column_names):
    using_fixture = getattr(processor, "_local_fixture_used", False)
    fixture_uri = getattr(processor, "_local_fixture_output_uri", None)

    if using_fixture and fixture_uri:
        parsed = urlparse(fixture_uri)
        if parsed.scheme == "file":
            output_df = pd.read_csv(parsed.path)
        elif parsed.scheme == "s3":
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            s3_client = boto3.client("s3", region_name=processor.sagemaker_session.boto_region_name)
            output_file_object = s3_client.get_object(Bucket=bucket, Key=key)
            output_df = pd.read_csv(io.BytesIO(output_file_object["Body"].read()))
        else:
            raise AssertionError(f"Unsupported fixture URI {fixture_uri}")
    else:
        s3_client = boto3.client("s3", region_name=processor.sagemaker_session.boto_region_name)
        key = f"{prefix}/{output_file_name}"
        output_file_object = s3_client.get_object(Bucket=bucket, Key=key)
        output_df = pd.read_csv(io.BytesIO(output_file_object["Body"].read()))
    assert all(col_name in output_df.columns for col_name in new_column_names)


def processing_job_completed(sagemaker_session, job_name):
    if not job_name:
        return False

    client = getattr(sagemaker_session, "sagemaker_client", None)
    if client is None:
        client = boto3.client("sagemaker", region_name=sagemaker_session.boto_region_name)

    response = client.describe_processing_job(ProcessingJobName=job_name)
    if not response or "ProcessingJobStatus" not in response:
        raise ValueError("Response is none or does not have ProcessingJobStatus")
    status = response["ProcessingJobStatus"]
    return status == "Completed"


def remove_test_resources(processor, bucket, prefix):
    if not bucket or not prefix:
        return
    region = processor.sagemaker_session.boto_region_name
    s3_resource = boto3.resource("s3", region_name=region)
    bucket_obj = s3_resource.Bucket(bucket) # type: ignore
    bucket_obj.objects.filter(Prefix=prefix).delete()
    processing_job_folder = processor._current_job_name
    if processing_job_folder:
        bucket_obj.objects.filter(Prefix=processing_job_folder).delete()
