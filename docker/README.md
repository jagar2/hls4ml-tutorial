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
> pass. For long sessions, refresh the token on the host and re-run. (The macOS
> Keychain value is `go-keyring-base64:`-prefixed base64 — decode it before use.)

## Training image (`Dockerfile.train`, linux/amd64)

The lightweight image above deliberately omits TensorFlow. To run the tutorial's
**notebook training** (and the `dp.keras_callback` tracking) in a container, use
`docker/Dockerfile.train`, which adds the CPU ML stack. The tutorial pins
TensorFlow 2.14 (x86-64 Linux wheels only), so it targets **linux/amd64** — native
on x86 hosts, emulated on Apple Silicon.

```bash
# amd64 daemon into the build context (git-ignored):
(cd cli && CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o ../docker/bin/dataerai-amd64 ./cmd/dataerai-transfer)

docker build --platform linux/amd64 -f docker/Dockerfile.train -t dataerai-hls4ml-train .

# run Part 1 training under papermill (captures provenance incl. training tracking):
docker run --rm --platform linux/amd64 \
  -e DATAERAI_SERVER=https://beta.dataerai.com \
  -e DATAERAI_PROJECT_ID=<your-project-uuid> \
  -v <host-creds>:/root/.config/dataerai/credentials:ro \
  dataerai-hls4ml-train run --only part1_getting_started
```

> On Apple Silicon this runs under emulation (slow); prefer native x86 for real
> training. To verify the tracking callback **without** the full toolchain, run a
> short Keras `fit()` with `dp.keras_callback(...)` in any TF environment and then
> `python run_pipeline.py refresh` — a training-log asset + a sealed lineage run
> land in the `hls4ml — <part>` collection.
