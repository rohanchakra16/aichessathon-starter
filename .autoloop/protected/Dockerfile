FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m pip install --no-cache-dir \
    chess==1.11.2 \
    mypy==2.3.1 \
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
