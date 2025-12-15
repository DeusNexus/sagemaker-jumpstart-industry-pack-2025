"""Test-time stubs for the SageMaker SDK when it is unavailable."""

from __future__ import annotations

import sys
import types
import uuid
from typing import Any, Dict, Iterable, List, Optional


def _ensure_sagemaker_stub() -> None:
    if "sagemaker" in sys.modules:
        return

    sagemaker_mod = types.ModuleType("sagemaker")
    core_mod = types.ModuleType("sagemaker.core")
    processing_mod = types.ModuleType("sagemaker.core.processing")
    network_mod = types.ModuleType("sagemaker.core.network")
    common_utils_mod = types.ModuleType("sagemaker.core.common_utils")

    class Session:
        def __init__(self, boto_session: Any = None, boto_region_name: str = "us-west-2"):
            self.boto_session = boto_session
            self.boto_region_name = boto_region_name

        def default_bucket(self) -> str:
            return "default-bucket"

        def process(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError("Stub Session does not implement process.")

    class Tags(List[Dict[str, str]]):
        """Placeholder container for tags."""

    class NetworkConfig:
        def __init__(self, **kwargs: Any):
            self.config = kwargs

    class ProcessingS3Input:
        def __init__(
            self,
            s3_uri: str,
            s3_data_type: str,
            local_path: Optional[str] = None,
            s3_input_mode: Optional[str] = None,
            s3_data_distribution_type: Optional[str] = None,
            s3_compression_type: Optional[str] = None,
        ):
            self.s3_uri = s3_uri
            self.s3_data_type = s3_data_type
            self.local_path = local_path
            self.s3_input_mode = s3_input_mode
            self.s3_data_distribution_type = s3_data_distribution_type
            self.s3_compression_type = s3_compression_type

    class ProcessingS3Output:
        def __init__(
            self,
            s3_uri: str,
            s3_upload_mode: str,
            local_path: Optional[str] = None,
        ):
            self.s3_uri = s3_uri
            self.s3_upload_mode = s3_upload_mode
            self.local_path = local_path

    class ProcessingInput:
        def __init__(
            self,
            *,
            input_name: str,
            app_managed: Optional[bool] = False,
            s3_input: Optional[ProcessingS3Input] = None,
            dataset_definition: Optional[Any] = None,
        ):
            self.input_name = input_name
            self.app_managed = bool(app_managed)
            self.s3_input = s3_input
            self.dataset_definition = dataset_definition

        def to_request_dict(self) -> Dict[str, Any]:
            data: Dict[str, Any] = {
                "InputName": self.input_name,
                "AppManaged": self.app_managed,
            }
            if self.s3_input is not None:
                data["S3Input"] = {
                    "S3Uri": self.s3_input.s3_uri,
                    "LocalPath": self.s3_input.local_path,
                    "S3DataType": self.s3_input.s3_data_type,
                    "S3InputMode": self.s3_input.s3_input_mode,
                    "S3DataDistributionType": self.s3_input.s3_data_distribution_type,
                    "S3CompressionType": self.s3_input.s3_compression_type,
                }
            return data

    class ProcessingOutput:
        def __init__(
            self,
            *,
            output_name: str,
            s3_output: Optional[ProcessingS3Output] = None,
            feature_store_output: Optional[Any] = None,
            app_managed: Optional[bool] = False,
        ):
            self.output_name = output_name
            self.s3_output = s3_output
            self.feature_store_output = feature_store_output
            self.app_managed = bool(app_managed)

        def to_request_dict(self, index: int) -> Dict[str, Any]:
            data: Dict[str, Any] = {
                "OutputName": self.output_name or f"output-{index}",
                "AppManaged": self.app_managed,
            }
            if self.s3_output is not None:
                data["S3Output"] = {
                    "S3Uri": self.s3_output.s3_uri,
                    "LocalPath": self.s3_output.local_path,
                    "S3UploadMode": self.s3_output.s3_upload_mode,
                }
            return data

    class Processor:
        def __init__(
            self,
            role: str,
            image_uri: str,
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
            self.role = role
            self.image_uri = image_uri
            self.instance_count = instance_count
            self.instance_type = instance_type
            self.volume_size_in_gb = volume_size_in_gb
            self.volume_kms_key = volume_kms_key
            self.output_kms_key = output_kms_key
            self.max_runtime_in_seconds = max_runtime_in_seconds
            self.sagemaker_session = sagemaker_session
            self.tags = tags
            self.base_job_name = base_job_name or "processor"
            self.network_config = network_config
            self._current_job_name: Optional[str] = None

        def run(
            self,
            inputs: Optional[Iterable[ProcessingInput]] = None,
            outputs: Optional[Iterable[ProcessingOutput]] = None,
            wait: bool = True,
            logs: bool = True,
        ):
            job_name = f"{self.base_job_name}-{uuid.uuid4().hex[:8]}"
            self._current_job_name = job_name

            request_inputs = [inp.to_request_dict() for inp in inputs or []]
            request_outputs = [
                output.to_request_dict(idx) for idx, output in enumerate(outputs or [], start=1)
            ]

            process_args = {
                "inputs": request_inputs,
                "output_config": {"Outputs": request_outputs},
                "experiment_config": None,
                "job_name": job_name,
                "resources": {
                    "ClusterConfig": {
                        "InstanceType": self.instance_type,
                        "InstanceCount": self.instance_count,
                        "VolumeSizeInGB": self.volume_size_in_gb,
                    }
                },
                "stopping_condition": None,
                "app_specification": {"ImageUri": self.image_uri},
                "environment": None,
                "network_config": self.network_config,
                "role_arn": self.role,
                "tags": self.tags,
            }

            if not self.sagemaker_session:
                raise RuntimeError("sagemaker_session is required for Processor.run()")
            return self.sagemaker_session.process(**process_args)

    def base_from_name(name: str) -> str:
        return name

    processing_mod.ProcessingS3Input = ProcessingS3Input  # type: ignore[attr-defined]
    processing_mod.ProcessingS3Output = ProcessingS3Output  # type: ignore[attr-defined]
    processing_mod.ProcessingInput = ProcessingInput  # type: ignore[attr-defined]
    processing_mod.ProcessingOutput = ProcessingOutput  # type: ignore[attr-defined]
    processing_mod.Processor = Processor  # type: ignore[attr-defined]
    processing_mod.Session = Session  # type: ignore[attr-defined]
    processing_mod.Tags = Tags  # type: ignore[attr-defined]

    network_mod.NetworkConfig = NetworkConfig  # type: ignore[attr-defined]
    common_utils_mod.base_from_name = base_from_name  # type: ignore[attr-defined]

    sagemaker_mod.core = core_mod  # type: ignore[attr-defined]
    core_mod.processing = processing_mod  # type: ignore[attr-defined]
    core_mod.network = network_mod  # type: ignore[attr-defined]
    core_mod.common_utils = common_utils_mod  # type: ignore[attr-defined]

    sys.modules["sagemaker"] = sagemaker_mod
    sys.modules["sagemaker.core"] = core_mod
    sys.modules["sagemaker.core.processing"] = processing_mod
    sys.modules["sagemaker.core.network"] = network_mod
    sys.modules["sagemaker.core.common_utils"] = common_utils_mod


_ensure_sagemaker_stub()
