FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The five packages the competition runtime preinstalls are torch 2.13.0+cpu,
# numpy 2.5.2, python-chess 1.11.2, onnxruntime 1.29.0 and numba 0.67.0
# (https://aichessathon.com/docs). This image mirrors the runtime for the parts
# an agent may legally use for speed: chess + numpy + numba (llvmlite is numba's
# hard dependency). torch and onnxruntime are intentionally omitted until a
# profiled need is shown and separately approved. mypy/pytest/ruff are the
# protected-suite dev tools and are not part of the runtime.
RUN python -m pip install --no-cache-dir \
    chess==1.11.2 \
    llvmlite==0.49.0 \
    mypy==2.3.1 \
    numba==0.67.0 \
    numpy==2.5.2 \
    pytest==9.1.1 \
    ruff==0.16.5

COPY . /image-build/submission
RUN if [ -f /image-build/submission/requirements.txt ]; then \
      python -m pip install --no-cache-dir --only-binary=:all: \
        --requirement /image-build/submission/requirements.txt; \
    fi \
    && rm -rf /image-build

WORKDIR /workspace
USER 10001:10001
