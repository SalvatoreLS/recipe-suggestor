# Model training

Trains the two YOLOv8 detectors the C++ assistant uses:

| Model | Dataset | Classes | Output |
|---|---|---|---|
| Bag of Crafting | `BoC---TBOI-6/` (300 train / 17 val) | 21 | `../recipe_suggestor_cpp/resources/models/boc_best.onnx` |
| Floor pickups | `pickups---TBOI-2/` (285 train / 19 val) | 23 | `../recipe_suggestor_cpp/resources/models/floor_best.onnx` |

The two class lists are **different**, and were both 21 classes until the floor dataset's v2 added
`black_heart` and `micro_battery`. Loading the wrong file produces confident nonsense rather than an
error, so the exported model's class names are checked against `class_map.json` at load time on the
C++ side — keep `floor.expects_classes` in step with `data.yaml`'s `names`, in the same order.

### Floor dataset: use v2, not v1

`pickups---TBOI-1/` is still on disk and should not be trained on. It has 85 train images / ~267
instances, and **no instances at all** for `golden_bomb` and `mega_battery` — two classes it cannot
possibly learn — plus six more with a single instance. v2 has 285 train images / 1446 instances and
covers every class.

The weak point in v2 is the validation split: only 14 of the 23 classes have validation instances at
all, so the headline number describes those 14 and says nothing about the other 9. `mega_battery`
and `red_soul_heart` have just 3 training instances each.

Result of the current `floor_best.onnx` (98 epochs, early-stopped, batch 8):
**mAP50 0.740 / mAP50-95 0.575** over the 14 evaluated classes.

| Strong (mAP50 > 0.9) | Weak | Not evaluated |
|---|---|---|
| dime 0.995, rotten_heart 0.995, key 0.971, penny 0.979, bomb 0.959, half_heart 0.914 | bone_heart 0.497, pill 0.497, golden_penny 0.497, half_soul_heart 0.538, card 0.796, double_bomb 0.858, red_heart 0.866 | battery, black_heart, golden_bomb, mega_battery, micro_battery, nickel, soul_heart, double_heart, red_soul_heart |

`eternal_heart` is the one outright failure: 4 validation instances, recall 0, **AP 0**. Treat
eternal-heart detection as not working — which matters, because it is one of the pool-forcing
pickups (Angel +10).

## Setup

```bash
pip install ultralytics roboflow python-dotenv pyyaml
```

> **Known issue:** `opencv-python` and `opencv-contrib-python` must not both be installed. When they
> are, uninstalling one deletes shared files the other owns and `cv2` degrades to an empty namespace
> package, which breaks `import ultralytics`. Keep one:
> `pip uninstall -y opencv-python opencv-contrib-python && pip install opencv-contrib-python`

## Training

Both scripts share `train_common.py`, so a fix applies to both.

```bash
# From a dataset already on disk -- no Roboflow API key needed
python3 train_floor.py --data pickups---TBOI-2/data.yaml
python3 train_boc.py   --data BoC---TBOI-6/data.yaml

# Or pull the dataset from Roboflow using .env
python3 train_floor.py
```

> `python3`, not `python` — there is no `python` on PATH on this machine.

> **Known issue:** the default batch size is **8**, not Ultralytics' 16. 16 does not fit in a 4 GB
> card — training runs for ~25 epochs and then dies with `torch.OutOfMemoryError: CUDA out of
> memory`. Raise `--batch` only if you have the VRAM for it. Even at 8 you may see
> `CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU` warnings; those are non-fatal fallbacks.
> Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation.

> **Known issue:** if training dies partway with
> `nvrtc: error: failed to open libnvrtc-builtins.so.13.0`, torch cannot find the JIT builtins it
> ships with. Point it at them first:
> `export LD_LIBRARY_PATH=~/.local/lib/python3.10/site-packages/nvidia/cu13/lib`
> Training runs for well over an hour, so start it detached (`setsid nohup … &`) rather than in a
> shell you might close.

Options: `--data`, `--epochs` (default 200), `--imgsz` (640), `--patience` (10), `--batch` (8),
`--device`, `--weights` (`yolov8n.pt`), `--name`.

### Choosing weights

```bash
python3 train_floor.py --data pickups---TBOI-3/data.yaml --weights yolo11s.pt
```

`--weights` takes a local file (resolved next to the script) or any name Ultralytics can fetch.
**yolo11s** is the recommended upgrade from the default `yolov8n`: 9.5M params against 3.0M, it fits
a 4 GB card at `--batch 8`, and its extra capacity helps on small pickup sprites.

It is safe for the C++ side because yolo11 keeps YOLOv8's detection head layout, `[1, 4+nc, 8400]`
with no objectness column and no built-in NMS. The `12` and `26` model families shipped with
Ultralytics have **not** been checked against the decoder — a different head would decode as
confident nonsense rather than fail, so the export step now verifies the ONNX output shape is
`[1, 4+nc, N]` and refuses to ship anything else.

Runs land in `runs/detect/<name>/`. The export always takes the weights from the run that just
finished, and refuses to export if the trained model's class names do not match the dataset — the
previous scripts hardcoded `runs/detect/train/weights/best.pt` and would happily have written a
**BoC** model into `floor_best.onnx`.

`.env` keys: `ROBOFLOW_API_KEY`, `ROBOFLOW_WORKSPACE`, `ROBOFLOW_PROJECT_BOC`,
`ROBOFLOW_VERSION_BOC`, `ROBOFLOW_PROJECT_FLOOR`, `ROBOFLOW_VERSION_FLOOR`.

## Extracting the real game data

`extract_game_data.py` recovers the collectible quality table and item pools from the installed
game, producing `../recipe_suggestor_cpp/resources/collectibles.json` (721 entries).

```bash
python3 extract_game_data.py                      # uses the default install path
python3 extract_game_data.py --game-dir /path/to/isaac
```

It runs the game's own bundled `tools/ResourceExtractor/Linux/resource_extractor`, unpacking to a
temp directory (~1.1 GB) that is deleted afterwards; `--extract-dir` plus `--skip-extract` reuse an
existing extraction instead.

Two things worth knowing:

- **Quality is not in `items.xml`.** That file has no `quality` attribute in v1.7.8a. It lives in
  `resources-dlc3/items_metadata.xml`. Pool membership comes from `resources-dlc3/itempools.xml`.
- **Names are not taken from the game.** `items.xml` stores localisation keys
  (`#THE_SAD_ONION_NAME`), so `resources/items.json` stays the name table and the script only
  cross-checks against it, reporting anything that disagrees.

Five ids in `items.json` (43, 61, 235, 587, 718) are blank, cut or unused, carry no metadata, and
are excluded — they are no longer craftable results.

## Export contract

The C++ side must match how the models were trained: 640×640 letterboxed, fixed `{1,3,640,640}`
input (`dynamic=False`), scale `1/255`, no mean subtraction, RGB, opset 12. The output is the
transposed YOLOv8 layout with no objectness column and no built-in NMS.

## Compatibility checks

- `cpp_compatibility_test/` — loads the exported ONNX in ONNX Runtime and runs one forward pass
- `java_compatibility_test/` — the DJL equivalent
- `test.py` — visual smoke test over `test_images/`, writing to `outputs/`
