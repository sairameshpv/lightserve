# lightserve's server, two flavours (docker build --target ...):
#   fake: no torch: the real scheduler with fake tokens (--fake-engine). Small, builds on any CPU
#         (the Mac's kind rehearsal is arm64).
#   gpu:  + CUDA 12.8 torch and Triton for the real engine on an L40S (amd64 only).
# Build from the repo root:  docker build -f infra/docker/lightserve.Dockerfile --target gpu -t lightserve .
FROM python:3.11-slim AS fake
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.141.1 uvicorn==0.52.4 prometheus_client==0.26.0 \
        opentelemetry-sdk==1.45.1 opentelemetry-exporter-otlp-proto-http==1.45.1
COPY engine/ engine/
COPY server/ server/
COPY model/ model/
COPY kernels/ kernels/
RUN useradd --uid 1000 lightserve
USER 1000
EXPOSE 8000
ENTRYPOINT ["python", "-m", "server"]
CMD ["--fake-engine"]

FROM fake AS gpu
USER root
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir safetensors==0.8.0 numpy
USER 1000
CMD []
