import sys
# Import Session from the path you specified
try:
    from sagemaker.core.processing import Session, LocalSession
except ImportError as e:
    print(f"Error: Could not import Session from sagemaker.core.processing. Please check your SageMaker SDK installation.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    sys.exit(1)

# Calling help() on an object in a script environment prints the documentation 
# directly to stdout, which the shell can then capture.
help(LocalSession)