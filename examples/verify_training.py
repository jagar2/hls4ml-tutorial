#!/usr/bin/env python
"""Smoke test for Dataerai neural-network training tracking.

Trains a tiny Keras CNN (mirroring part6's pruned-CNN training call) with
``dp.keras_callback`` and confirms the full path is recorded:

* a **training-log** asset (hyperparameters + per-epoch metrics),
* the **trained model** asset (linked ``training-log --derived_from--> model``),
* a ``trained_on`` edge to a referenced **dataset** asset, and
* on ``dp.refresh`` — the async DIDs + the **sealed lineage run** (integrity root).

Offline by default (no credentials needed)::

    DATAERAI_DRY_RUN=1 python examples/verify_training.py

Against beta (creds + owner project)::

    dataerai auth login --server https://beta.dataerai.com
    DATAERAI_PROJECT_ID=<uuid> python examples/verify_training.py

Requires TensorFlow (any recent version); it is only imported when run.
"""
import os
import sys

# Offline unless credentials + an owner project are configured: the integration
# auto-degrades to dry-run when no token is found (or force it with DATAERAI_DRY_RUN=1).
os.environ.setdefault("DATAERAI_SERVER", "https://beta.dataerai.com")

NOTEBOOK = "part6_cnns"


def main() -> int:
    try:
        import numpy as np
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print("TensorFlow is required for this smoke test — skipping.", file=sys.stderr)
        return 0

    import dataerai_hls4ml as dp

    run = dp.get_run("hls4ml-pipeline", kind="training", fresh=True)
    run.use_collection(f"hls4ml — {NOTEBOOK}")

    # 1) reference the external (tfds) dataset so the model can be trained_on it
    dataset = dp.reference_dataset(
        "SVHN (svhn_cropped)", source="tfds:svhn_cropped",
        metadata={"classes": 10, "input_shape": [32, 32, 3],
                  "splits": "train[:90%] / train[-10%:] / test"})

    # 2) a tiny CNN mirroring part6's pruned-CNN structure, on random SVHN-shaped data
    rng = np.random.default_rng(0)
    X = rng.random((256, 32, 32, 3), dtype="float32")
    Y = keras.utils.to_categorical(rng.integers(0, 10, 256), 10)
    Xv = rng.random((64, 32, 32, 3), dtype="float32")
    Yv = keras.utils.to_categorical(rng.integers(0, 10, 64), 10)
    model = keras.Sequential([
        layers.Input((32, 32, 3)),
        layers.Conv2D(8, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(8, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dense(16, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(keras.optimizers.Adam(1e-3), "categorical_crossentropy", metrics=["accuracy"])

    # 3) exactly the wiring pattern used in part6's training cells
    callbacks = [keras.callbacks.EarlyStopping(patience=3)]
    callbacks = list(callbacks) + [dp.keras_callback(
        "Pruned CNN (part6) training", notebook=NOTEBOOK,
        model_title="Pruned CNN (part6)", dataset=dataset)]
    model.fit(X, Y, validation_data=(Xv, Yv), epochs=2, batch_size=64,
              callbacks=callbacks, verbose=2)

    # 4) seal: re-fetch async-minted DIDs + close the lineage run
    man = dp.refresh(".")

    mode = "dry-run" if man["dry_run"] else man["server"]
    print(f"\nRUN {man['run_id']}  mode={mode}  artifacts={len(man['artifacts'])}  "
          f"edges={len(man['edges'])}  citations={len(man.get('citations') or [])}  "
          f"integrity={man.get('integrity_root')}")
    for a in man["artifacts"]:
        print(f"  {a['kind']:13s} {a['name']:40s} {a['asset_id']}")
    kinds = {a["kind"] for a in man["artifacts"]}
    edge_types = {e["type"] for e in man["edges"]}
    ok = {"training_log", "model", "dataset"} <= kinds and {"derived_from", "trained_on"} <= edge_types
    print(f"\nsmoke test: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
