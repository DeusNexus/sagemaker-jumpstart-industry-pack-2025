# Copyright Amazon.com, Inc. or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
SageMaker JumpStart Industry processing job module (SDK v3 compatible).
"""

import copy
import json
import logging
import os
import shutil
import tempfile
import uuid
from typing import Dict, Mapping, MutableMapping, Optional, Union
from urllib.parse import urlparse

JSONDict = MutableMapping[str, object]

try:
    import boto3  # type: ignore[import]
except ImportError:  # pragma: no cover - handled via tests
    boto3 = None  # type: ignore[assignment]

try:
    from sagemaker.core.processing import (
        ProcessingInput,
        ProcessingOutput,
        ProcessingS3Input,
        ProcessingS3Output,
        Processor,
        Session,
        Tags,
    )
except ImportError:
    from sagemaker.core.processing import (
        ProcessingInput,
        ProcessingOutput,
        Processor,
        Session,
        Tags,
    )
    from sagemaker.core.shapes.shapes import ProcessingS3Input, ProcessingS3Output
from sagemaker.core.network import NetworkConfig
from sagemaker.core.utils.exceptions import FailedStatusError
from sagemaker.core.common_utils import base_from_name

from smjsindustry.finance.processor_config import (
    JaccardSummarizerConfig,
    KMedoidsSummarizerConfig,
    NLPScorerConfig,
    EDGARDataSetConfig,
)
from smjsindustry.finance.constants import (
    SUMMARIZER_JOB_NAME,
    NLP_SCORE_JOB_NAME,
    JACCARD_SUMMARIZER,
    SEC_FILING_PARSER_JOB_NAME,
    SEC_XML_FILING_PARSER,
    SEC_FILING_RETRIEVAL_JOB_NAME,
)
from smjsindustry.finance.utils import retrieve_image

LOCAL_DATALOADER_FIXTURE_ENV = "SMJS_FINANCE_DATALOADER_LOCAL_DATASET"
LOCAL_DATALOADER_FALLBACK_ENV = "SMJS_FINANCE_DATALOADER_FALLBACK_DATASET"

logger = logging.getLogger(__name__)


# =============================================================================
# Base processor
# =============================================================================

class FinanceProcessor(Processor):
    """Base class for all JumpStart Industry processing jobs."""

    _PROCESSING_CONFIG = "/opt/ml/processing/input/config"
    _PROCESSING_DATA = "/opt/ml/processing/input/data"
    _PROCESSING_OUTPUT = "/opt/ml/processing/output"
    _DEFAULT_OUTPUT_NAME = "output-1"

    _CONFIG_FILE = "job_config.json"
    _CONFIG_INPUT_NAME = "config"
    _DATA_INPUT_NAME = "data"

    def __init__(
        self,
        role: str,
        instance_count: int,
        instance_type: str,
        volume_size_in_gb: int = 30,
        volume_kms_key: Optional[str] = None,
        output_kms_key: Optional[str] = None,
        max_runtime_in_seconds: Optional[int] = None,
        sagemaker_session: Optional[Session] = None,
        tags: Optional[Tags] = None,
        base_job_name: Optional[str] = None,
        network_config: Optional[NetworkConfig] = None,
    ):
        if sagemaker_session is None:
            session: Session = Session()
        else:
            session = sagemaker_session

        region = session.boto_region_name
        if region is None:
            raise ValueError("SageMaker session must have an associated region.")

        container_uri = retrieve_image(region)

        if boto3 is None:
            raise RuntimeError(
                "boto3 is required to use FinanceProcessor. "
                "Install boto3 or provide a compatible stub."
            )

        super().__init__(
            role=role,
            image_uri=container_uri,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            sagemaker_session=session,
            tags=tags,
            base_job_name=base_job_name,
            network_config=network_config,
        )

        self._s3 = boto3.client(
            "s3",
            region_name=region,
        )

    def _build_processing_input(self, input_name: str, source: str, destination: str) -> ProcessingInput:
        """Create a ProcessingInput definition aligned with SageMaker SDK v3."""
        return ProcessingInput(
            input_name=input_name,
            s3_input=ProcessingS3Input(
                s3_uri=source,
                local_path=destination,
                s3_data_type="S3Prefix",
                s3_input_mode="File",
                s3_data_distribution_type="FullyReplicated",
                s3_compression_type="None",
            ),
        )

    def _build_processing_output(self, s3_destination: str) -> ProcessingOutput:
        """Create a ProcessingOutput definition aligned with SageMaker SDK v3."""
        return ProcessingOutput(
            output_name=self._DEFAULT_OUTPUT_NAME,
            s3_output=ProcessingS3Output(
                s3_uri=s3_destination,
                local_path=self._PROCESSING_OUTPUT,
                s3_upload_mode="EndOfJob",
            ),
        )

    # ---------------------------------------------------------------------
    # S3 helpers (REQUIRED for SDK v3)
    # ---------------------------------------------------------------------

    def _upload_dir_to_s3(self, local_dir: str, s3_prefix: str) -> str:
        parsed = urlparse(s3_prefix)
        if parsed.scheme != "s3":
            raise ValueError("s3_prefix must be an s3:// URI")

        bucket = parsed.netloc
        key_prefix = parsed.path.lstrip("/")

        for root, _, files in os.walk(local_dir):
            for fname in files:
                local_path = os.path.join(root, fname)
                rel_path = os.path.relpath(local_path, local_dir)
                s3_key = f"{key_prefix}/{rel_path}"
                self._s3.upload_file(local_path, bucket, s3_key)

        return f"s3://{bucket}/{key_prefix}"

    def _upload_file_to_s3_uri(self, local_path: str, s3_uri: str) -> str:
        parsed = urlparse(s3_uri)
        if parsed.scheme != "s3":
            raise ValueError("Destination must be an s3:// URI")
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not os.path.isfile(local_path):
            raise FileNotFoundError(local_path)
        self._s3.upload_file(local_path, bucket, key)
        return f"s3://{bucket}/{key}"

    def _ensure_s3_input(self, path: str, s3_base_prefix: str) -> str:
        if path.startswith("s3://"):
            return path

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        upload_prefix = f"{s3_base_prefix}/{uuid.uuid4().hex}"
        parsed = urlparse(upload_prefix)

        if os.path.isfile(path):
            bucket = parsed.netloc
            key = f"{parsed.path.lstrip('/')}/{os.path.basename(path)}"
            self._s3.upload_file(path, bucket, key)
            return f"s3://{bucket}/{key}"

        return self._upload_dir_to_s3(path, upload_prefix)

    # ---------------------------------------------------------------------

    def run(self, *args, **kwargs):
        logger.info(
            "EC2 instances are not billed while in the 'pending' state. "
            "See: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-lifecycle.html"
        )
        return super().run(*args, **kwargs)


# =============================================================================
# Summarizer
# =============================================================================

class Summarizer(FinanceProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            base_job_name=base_from_name(SUMMARIZER_JOB_NAME),
            **kwargs,
        )

    def summarize(
        self,
        summarizer_config: Union[JaccardSummarizerConfig, KMedoidsSummarizerConfig],
        text_column_name: str,
        input_file_path: str,
        s3_output_path: str,
        output_file_name: str,
        new_summary_column_name: str = "summary",
        wait: bool = True,
        logs: bool = True,
    ):
        if urlparse(s3_output_path).scheme != "s3":
            raise ValueError("s3_output_path must be an s3:// URI")

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, self._CONFIG_FILE)

            cfg: JSONDict = dict(copy.deepcopy(summarizer_config.get_config()))

            vocab = cfg.get("vocabulary")

            if cfg.get("processor_type") == JACCARD_SUMMARIZER and isinstance(vocab, set):
                cfg["vocabulary"] = list(vocab)

            cfg.update(
                text_column_name=text_column_name,
                new_summary_column_name=new_summary_column_name,
                output_file_name=output_file_name,
            )

            with open(cfg_path, "w") as f:
                json.dump(cfg, f)

            s3_cfg = self._upload_dir_to_s3(tmp, f"{s3_output_path}/_config")
            s3_data = self._ensure_s3_input(input_file_path, f"{s3_output_path}/_data")

            inputs = [
                self._build_processing_input(self._CONFIG_INPUT_NAME, s3_cfg, self._PROCESSING_CONFIG),
                self._build_processing_input(self._DATA_INPUT_NAME, s3_data, self._PROCESSING_DATA),
            ]

            outputs = [self._build_processing_output(s3_output_path)]

            logger.info("Starting summarization job")
            self.run(inputs=inputs, outputs=outputs, wait=wait, logs=logs)


# =============================================================================
# NLP Scorer
# =============================================================================

class NLPScorer(FinanceProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            base_job_name=base_from_name(NLP_SCORE_JOB_NAME),
            **kwargs,
        )

    def calculate(
        self,
        score_config: NLPScorerConfig,
        text_column_name: str,
        input_file_path: str,
        s3_output_path: str,
        output_file_name: str,
        wait: bool = True,
        logs: bool = True,
    ):
        if urlparse(s3_output_path).scheme != "s3":
            raise ValueError("s3_output_path must be an s3:// URI")

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, self._CONFIG_FILE)

            cfg = copy.deepcopy(score_config.get_config())
            cfg.update(
                text_column_name=text_column_name,
                output_file_name=output_file_name,
            )

            with open(cfg_path, "w") as f:
                json.dump(cfg, f)

            s3_cfg = self._upload_dir_to_s3(tmp, f"{s3_output_path}/_config")
            s3_data = self._ensure_s3_input(input_file_path, f"{s3_output_path}/_data")

            self.run(
                inputs=[
                    self._build_processing_input(self._CONFIG_INPUT_NAME, s3_cfg, self._PROCESSING_CONFIG),
                    self._build_processing_input(self._DATA_INPUT_NAME, s3_data, self._PROCESSING_DATA),
                ],
                outputs=[self._build_processing_output(s3_output_path)],
                wait=wait,
                logs=logs,
            )


# =============================================================================
# Data Loader
# =============================================================================

class DataLoader(FinanceProcessor):

    def __init__(self, *args, **kwargs):
        # EDGAR retrieval must run on a single instance
        if kwargs.get("instance_count", 1) != 1:
            logger.info(
                "DataLoader only supports instance_count=1; overriding value."
            )
            kwargs["instance_count"] = 1

        super().__init__(
            *args,
            base_job_name=SEC_FILING_RETRIEVAL_JOB_NAME,
            **kwargs,
        )
        self._local_fixture_used = False
        self._local_fixture_output_uri: Optional[str] = None

    def load(
        self,
        dataset_config: EDGARDataSetConfig,
        s3_output_path: str,
        output_file_name: str,
        wait: bool = True,
        logs: bool = True,
    ):
        parsed_output = urlparse(s3_output_path)
        self._local_fixture_used = False
        self._local_fixture_output_uri = None
        local_fixture_path = os.getenv(LOCAL_DATALOADER_FIXTURE_ENV)
        fallback_fixture_path = os.getenv(LOCAL_DATALOADER_FALLBACK_ENV)

        if parsed_output.scheme != "s3" and not (
            local_fixture_path and parsed_output.scheme in ("file", "")
        ):
            raise ValueError("s3_output_path must be an s3:// URI")

        if local_fixture_path:
            logger.info(
                "Detected %s; uploading local fixture '%s' instead of running a "
                "remote DataLoader job.",
                LOCAL_DATALOADER_FIXTURE_ENV,
                local_fixture_path,
            )
            self._run_local_fixture(local_fixture_path, s3_output_path, output_file_name)
            self._local_fixture_used = True
            return

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, self._CONFIG_FILE)

            cfg: dict[str, object] = dict(
                copy.deepcopy(dataset_config.get_config())
            )
            cfg["output_file_name"] = output_file_name

            with open(cfg_path, "w") as f:
                json.dump(cfg, f)

            s3_cfg = self._upload_dir_to_s3(tmp, f"{s3_output_path}/_config")

            try:
                self.run(
                    inputs=[
                        self._build_processing_input(
                            self._CONFIG_INPUT_NAME, s3_cfg, self._PROCESSING_CONFIG
                        )
                    ],
                    outputs=[self._build_processing_output(s3_output_path)],
                    wait=wait,
                    logs=logs,
                )
            except FailedStatusError as exc:
                if fallback_fixture_path:
                    logger.warning(
                        "DataLoader job failed (%s). Falling back to local dataset '%s'",
                        exc,
                        fallback_fixture_path,
                    )
                    self._run_local_fixture(
                        fallback_fixture_path, s3_output_path, output_file_name
                    )
                    self._local_fixture_used = True
                    return
                raise

    def _run_local_fixture(self, fixture_path: str, output_uri: str, output_file_name: str) -> None:
        parsed = urlparse(output_uri)
        if parsed.scheme == "s3":
            destination_uri = f"{output_uri.rstrip('/')}/{output_file_name}"
            self._upload_file_to_s3_uri(fixture_path, destination_uri)
            logger.info("Uploaded dataloader fixture to %s", destination_uri)
        elif parsed.scheme in ("file", ""):
            base_path = parsed.path or output_uri
            os.makedirs(base_path, exist_ok=True)
            destination_path = os.path.join(base_path, output_file_name)
            shutil.copyfile(fixture_path, destination_path)
            destination_uri = f"file://{destination_path}"
            logger.info("Copied dataloader fixture to %s", destination_path)
        else:
            raise ValueError("Fixture output path must be an s3:// or file:// URI")

        self._local_fixture_output_uri = destination_uri


# =============================================================================
# SEC XML Filing Parser
# =============================================================================

class SECXMLFilingParser(FinanceProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            base_job_name=SEC_FILING_PARSER_JOB_NAME,
            **kwargs,
        )

    def parse(
        self,
        input_data_path: str,
        s3_output_path: str,
        wait: bool = True,
        logs: bool = True,
    ):
        if urlparse(s3_output_path).scheme != "s3":
            raise ValueError("s3_output_path must be an s3:// URI")

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, self._CONFIG_FILE)

            with open(cfg_path, "w") as f:
                json.dump({"processor_type": SEC_XML_FILING_PARSER}, f)

            s3_cfg = self._upload_dir_to_s3(tmp, f"{s3_output_path}/_config")
            s3_data = self._ensure_s3_input(input_data_path, f"{s3_output_path}/_data")

            self.run(
                inputs=[
                    self._build_processing_input(self._CONFIG_INPUT_NAME, s3_cfg, self._PROCESSING_CONFIG),
                    self._build_processing_input(self._DATA_INPUT_NAME, s3_data, self._PROCESSING_DATA),
                ],
                outputs=[self._build_processing_output(s3_output_path)],
                wait=wait,
                logs=logs,
            )
