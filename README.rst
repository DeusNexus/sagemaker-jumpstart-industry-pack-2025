=======================================
SageMaker JumpStart Industry Python SDK
=======================================

.. inclusion-marker-1-starting-do-not-remove

.. image:: https://img.shields.io/pypi/v/smjsindustry.svg
   :target: https://pypi.python.org/pypi/smjsindustry
   :alt: Latest Version

.. image:: https://img.shields.io/pypi/pyversions/smjsindustry.svg
   :target: https://pypi.python.org/pypi/smjsindustry
   :alt: Supported Python Versions

.. image:: https://readthedocs.org/projects/sagemaker-jumpstart-industry-pack/badge/?version=latest
   :target: https://sagemaker-jumpstart-industry-pack.readthedocs.io/en/latest/
   :alt: Documentation Status

The SageMaker JumpStart Industry Python SDK is a client library of `Amazon
SageMaker JumpStart <https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html>`_.
The library provides tools for feature engineering, training, and
deploying industry-focused machine learning
models on SageMaker JumpStart. With this industry-focused SDK,
you can curate text datasets, and train and deploy
language models.

The SDK now targets the SageMaker Python SDK v3 API surface and Pydantic v2
validation runtime. All processors and configuration objects have been
updated accordingly.

.. inclusion-marker-1-ending-do-not-remove

.. inclusion-marker-1-1-starting-do-not-remove

In particular, for the financial services industry, you can use a new set of
multimodal (long-form text, tabular) financial analysis tools within Amazon
SageMaker JumpStart. With these new tools, you can enhance your tabular ML
workflows with new insights from financial text documents and help save weeks
of development time. By using the SDK, you can directly retrieve financial documents
such as SEC filings, and further process financial text documents with
features such as summarization and scoring for sentiment, litigiousness,
risk, and readability.

In addition, you can access language models pretrained
on financial texts for transfer learning, and use example notebooks for data
retrieval, feature engineering of text data, enhancing the data into multimodal datasets,
and improve model performance.

SageMaker JumpStart Industry also provides prebuilt solutions for specific use cases
(for example, credit scoring), which are fully customizable and showcase the use of
AWS CloudFormation templates and reference architectures to accelerate your
machine learning journey.

.. inclusion-marker-1-1-ending-do-not-remove

For detailed documentation, including the API reference,
see `ReadTheDocs <https://sagemaker-jumpstart-industry-pack.readthedocs.io/en/latest/>`_.

.. inclusion-marker-2-starting-do-not-remove


Installing the SageMaker JumpStart Industry Python SDK
------------------------------------------------------

The SageMaker JumpStart Industry Python SDK is released to PyPI and
can be installed with pip as follows:

.. code-block:: bash

    pip install smjsindustry


You can also install from source by cloning this repository and running
a pip install command in the root directory of the repository:

.. code-block:: bash

    git clone https://github.com/aws/sagemaker-jumpstart-industry-python-sdk.git
    cd sagemaker-jumpstart-industry-python-sdk
    pip install .


Supported Operating Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SageMaker JumpStart Industry Python SDK supports Unix/Linux and Mac.

Supported Python Versions
~~~~~~~~~~~~~~~~~~~~~~~~~

The SageMaker JumpStart Industry Python SDK is tested on:

- Python 3.11
- Python 3.12


AWS Permissions
~~~~~~~~~~~~~~~

The SageMaker JumpStart Industry Python SDK runs on Amazon SageMaker. As a managed service, Amazon SageMaker performs operations on your behalf
on the AWS hardware that is managed by Amazon SageMaker.
Amazon SageMaker can perform only operations that the user permits.
You can read more about which permissions are necessary in the
`Amazon SageMaker Documentation
<https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html>`__.

The SageMaker JumpStart Industry Python SDK should not require any additional permissions aside from what is required for using SageMaker.
However, if you are using an IAM role with a path in it, you should grant permission for ``iam:GetRole``.

AWS Account Setup
~~~~~~~~~~~~~~~~~

To reproduce the integration tests and example workflows we recommend the following
account configuration (verified in the ``us-east-1`` region):

1. Create an IAM role named ``SageMakerRole`` (case-sensitive). The role should:

   - Trust the ``sagemaker.amazonaws.com`` service.
   - Include the AWS-managed ``AmazonSageMakerFullAccess`` policy.
   - Include ``AmazonS3FullAccess`` (or a scoped bucket policy covering the buckets you use).
   - Include ``AmazonEC2ContainerRegistryFullAccess`` (or a scoped policy covering the JumpStart ECR accounts).
   - Include ``CloudWatchLogsFullAccess`` so processing jobs can emit logs.
   - Include ``ElasticInferenceFullAccess`` if you plan to run EI-enabled models (see the `Elastic Inference setup guide <https://docs.aws.amazon.com/sagemaker/latest/dg/ei-setup.html>`_).
   - Allow ``iam:GetRole`` and ``sts:AssumeRole`` so SDK helpers can resolve the ARN.

2. Make sure your AWS credentials (user or federated role) permit you to ``sts:AssumeRole`` into ``SageMakerRole`` and have standard SageMaker prerequisites (e.g., ``s3:CreateBucket`` if you rely on the default SageMaker bucket).

3. If you operate in additional regions, replicate the same role name/policy set so the CLI helpers keep working across regions.

4. Configure your local AWS CLI/SDK credentials (for example via ``aws configure``) so the SDK can discover ``~/.aws/credentials`` and ``~/.aws/config``. At a minimum, specify:

   - ``aws_access_key_id`` and ``aws_secret_access_key`` for an IAM user/role with permission to assume ``SageMakerRole``.
   - ``default region name`` (e.g., ``us-east-1``) so SageMaker sessions inherit the correct region.


Licensing
~~~~~~~~~
The SageMaker JumpStart Industry Python SDK is licensed
under the Apache 2.0 License.
It is copyright Amazon.com, Inc. or its affiliates.
All Rights Reserved. The license is available at
`Apache License <http://aws.amazon.com/apache2.0/>`_.


Legal Notes
~~~~~~~~~~~

1. The SageMaker JumpStart Industry solutions, notebooks, demos, and examples are for demonstrative purposes only. It is not financial advice and should not be relied on as financial or investment advice.
2. The SageMaker JumpStart Industry solutions, notebooks, demos, and examples
   use data obtained from the SEC EDGAR database. You are responsible for complying
   with EDGAR’s access terms and conditions located in the
   `Accessing EDGAR Data <https://www.sec.gov/os/accessing-edgar-data>`_ page.


Running Tests
~~~~~~~~~~~~~

The SageMaker JumpStart Industry SDK has unit tests and integration tests.

You can install the libraries needed to run the tests by running :code:`pip install --upgrade .[test]` or, for Zsh users: :code:`pip install --upgrade .\[test\]`

**Unit tests**

We use tox to run Unit tests. Tox is an automated test tool that helps you run unit tests easily on multiple Python versions, and also checks the
code sytle meets our standards. We run tox with all of our supported Python versions(Python 3.11, Python 3.12). In order to run unit tests
with the same configuration as we do, you need to have interpreters for those Python versions installed.

To run the unit tests with tox, run:

::

    tox tests/unit

**Integrations tests**

To run the integration tests, you need to first prepare an AWS account with certain configurations:

1. AWS account credentials are available in the environment for the boto3 client to use.
2. The AWS account has an IAM role named :code:`SageMakerRole`.
   It should have the AmazonSageMakerFullAccess policy attached as well as a policy with `the necessary permissions to use Elastic Inference <https://docs.aws.amazon.com/sagemaker/latest/dg/ei-setup.html>`__.

We recommend selectively running just those integration tests you would like to run. You can filter by individual test function names with:

::

    tox -- -k 'test_function_i_care_about'


You can also run all of the integration tests by running the following command, which runs them in sequence, which may take a while:

::

    tox -- tests/integ


SEC Test Configuration
~~~~~~~~~~~~~~~~~~~~~~

Several integration tests interact with the SEC EDGAR service. Configure your shell so
every request contains a valid contact email and (optionally) so tests can run entirely
offline:

.. code-block:: bash

    # Required: SEC user-agent string with contact information
    export SMJS_FINANCE_SEC_USER_AGENT="MyPipeline/1.0 (contact: you@example.com)"

    # Optional: skip STS lookups if you know the execution-role ARN
    export SMJS_FINANCE_EXECUTION_ROLE_ARN="arn:aws:iam::<account-id>:role/SageMakerRole"

    # Optional: supply a local CSV so DataLoader tests bypass SEC download/SageMaker
    export SMJS_FINANCE_DATALOADER_LOCAL_DATASET="tests/data/finance/sec_dataloader_fixture.csv"

    # Optional: attempt the live SEC download but fall back to a local CSV if it fails
    export SMJS_FINANCE_DATALOADER_FALLBACK_DATASET="tests/data/finance/sec_dataloader_fixture.csv"

The repository also includes ``tests/data/finance/ticker.txt``. Use this file (or your
own pre-downloaded ticker list) when preparing fixtures that need the SEC ticker/CIK
mapping.

Example commands:

.. code-block:: bash

    # Run unit tests (tox manages interpreters)
    tox tests/unit

    # Run the dataloader integration test using a local fixture (no SageMaker job)
    . .venv/bin/activate
    SMJS_FINANCE_SEC_USER_AGENT="MyPipeline/1.0 (contact: you@example.com)" \
    SMJS_FINANCE_EXECUTION_ROLE_ARN="arn:aws:iam::<account-id>:role/SageMakerRole" \
    SMJS_FINANCE_DATALOADER_LOCAL_DATASET="tests/data/finance/sec_dataloader_fixture.csv" \
    PYTHONPATH=src pytest tests/integ/finance/test_finance.py::test_dataloader -s

    # Run the full integration suite (requires AWS + SEC connectivity)
    . .venv/bin/activate
    SMJS_FINANCE_SEC_USER_AGENT="MyPipeline/1.0 (contact: you@example.com)" \
    PYTHONPATH=src pytest tests/integ -s


Building Sphinx Docs Locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the dev version of the library:

.. code-block::

    pip install -e .\[all\]

Install Sphinx and the dependencies listed in ``sagemaker-jumpstart-industry-python-sdk/docs/requirements.txt``:

.. code-block::

    pip install sphinx
    pip install -r sagemaker-jumpstart-industry-python-sdk/docs/requirements.txt

Then ``cd`` into the ``sagemaker-jumpstart-industry-python-sdk/docs`` directory and run:

.. code-block::

    make html && open build/html/index.html


.. inclusion-marker-2-ending-do-not-remove
