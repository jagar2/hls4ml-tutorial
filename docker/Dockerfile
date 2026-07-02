# Dataerai × hls4ml — data-preservation & provenance runtime (beta-ready).
#
# Lightweight image containing the Dataerai Python SDK, the `dataerai_hls4ml`
# integration, and the Dataerai transfer daemon. It exercises the *preservation +
# provenance* path (upload artifacts as versioned assets, build the lineage DAG,
# resolve DID citations) against any Dataerai backend, including beta.
#
# TensorFlow / hls4ml are intentionally NOT installed here (no linux/arm64 TF 2.14
# wheels; synthesis needs Vitis anyway) — model training/synthesis run in the full
# conda env from environment.yml. See docker/README.md.
#
# Build context = repo root:  docker build -f docker/Dockerfile -t dataerai-hls4ml .
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dataerai transfer daemon (Linux). Place your platform's binary at
# docker/bin/dataerai before building — see docker/README.md.
COPY docker/bin/dataerai /usr/local/bin/dataerai
RUN chmod +x /usr/local/bin/dataerai
ENV DATAERAI_BINARY=/usr/local/bin/dataerai

# Dataerai Python SDK (preservation via the daemon).
COPY docker/vendor/dataerai-sdk /opt/dataerai-sdk
RUN pip install --no-cache-dir /opt/dataerai-sdk

# The integration + tutorial sources.
COPY . /app
RUN pip install --no-cache-dir -e .

ENV DATAERAI_SERVER=https://beta.dataerai.com
ENTRYPOINT ["python", "run_pipeline.py"]
CMD ["selftest", "--dry-run"]
