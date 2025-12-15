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
"""Central configuration helpers for SageMaker JumpStart Industry."""

import os
from typing import Optional

SEC_USER_AGENT_ENV = "SMJS_FINANCE_SEC_USER_AGENT"
LEGACY_SEC_CONTACT_EMAIL_ENV = "SMJS_FINANCE_SEC_CONTACT_EMAIL"
# Optional in-repo default to avoid setting the env var locally.
# Replace this with your real SEC-compliant user agent string, e.g.
# ``"MyDataPipeline/1.0 (contact: your.name@yourdomain.com)"``.
DEFAULT_SEC_USER_AGENT: Optional[str] = "ExampleDataPipeline/1.0 (contact: contact@example.com)"


def get_env_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    """Fetch an environment variable, returning ``default`` if unset."""

    return os.environ.get(name, default)


def get_sec_user_agent(default: Optional[str] = None) -> Optional[str]:
    """Return the SEC-compliant user agent string configured for EDGAR access."""

    if default is None:
        default = DEFAULT_SEC_USER_AGENT
    return get_env_setting(
        SEC_USER_AGENT_ENV,
        get_env_setting(LEGACY_SEC_CONTACT_EMAIL_ENV, default),
    )
