# Containerised Dataerai provenance runtime

A lightweight image to exercise the Dataerai **preservation + provenance** path
(including against **beta**) without the heavy TensorFlow/Vitis toolchain. Model
training & FPGA synthesis run in the full conda env (`environment.yml`); this
image is for the Dataerai integration.

## Prerequisites (build context)

Two files are arch-specific / proprietary, so you supply them (both git-ignored):

```bash
# 1) The Dataerai transfer daemon for the image's platform (linux/<arch>).
#    Build it from the dataerai monorepo, e.g. for linux/arm64:
#    (cd cli && CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -o dataerai ./cmd/dataerai-transfer)
mkdir -p docker/bin && cp /path/to/linux/dataerai docker/bin/dataerai

# 2) The Dataerai Python SDK source (from the monorepo).
mkdir -p docker/vendor && cp -r /path/to/datatransfer-dataerai/sdk/python docker/vendor/dataerai-sdk
```

## Build

```bash
docker build -f docker/Dockerfile -t dataerai-hls4ml .
```

## Run — offline (no credentials)

```bash
docker run --rm dataerai-hls4ml                      # selftest --dry-run (full synthetic DAG)
docker run --rm dataerai-hls4ml capture --dry-run    # capture the repo's example dirs
```

## Run — against beta (real assets + lineage)

Authenticate on your host first (opens a browser or prints a device code):

```bash
dataerai auth login --server https://beta.dataerai.com
dataerai auth login --device        # headless alternative
```

Then pass the token + owner project into the container. Export the token from the
host credential store:

```bash
TOKEN=$(security find-generic-password -s dataerai -a auth -w | python -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')   # macOS
# (Linux: read ~/.config/dataerai/credentials)

docker run --rm \
  -e DATAERAI_SERVER=https://beta.dataerai.com \
  -e DATAERAI_ACCESS_TOKEN="$TOKEN" \
  -e DATAERAI_PROJECT_ID=<your-project-uuid> \
  dataerai-hls4ml capture --root /app
```

Assets appear under your beta project; the lineage run id + edges are printed and
written to `provenance_manifest.json`. See [../DATAERAI_PROVENANCE.md](../DATAERAI_PROVENANCE.md).

> The daemon in the container performs the S3 multipart upload using the token you
> pass. For long sessions, refresh the token on the host and re-run.
