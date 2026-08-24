"""Shared training/export logic for the BoC and Floor YOLOv8 models.

Both models are trained identically; only the dataset and the destination
filename differ. Keeping the logic here means a fix lands in both at once --
the previous per-script copies had diverged into a silent stale-export bug.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE.parent / "recipe_suggestor_cpp" / "resources" / "models"
# Starting weights. A bare filename is resolved next to this script if it is
# already downloaded, otherwise Ultralytics fetches it by name. yolo11s is the
# recommended upgrade: it keeps the same detection head layout as yolov8
# ([1, 4+nc, 8400]), so the C++ decoder needs no change, and its extra capacity
# helps on small pickup sprites.
BASE_WEIGHTS = "yolov8n.pt"

IMG_SIZE = 640
EPOCHS = 200
PATIENCE = 10
# Ultralytics defaults to 16, which does not fit in a 4 GB card: training dies
# with a CUDA OutOfMemoryError partway through (seen at epoch 26 on a Quadro
# T2000). 8 fits with room to spare.
BATCH = 8

load_dotenv(HERE / ".env")


def expected_classes(data_yaml):
    """The class list the trained model must report, taken from the dataset."""
    with open(data_yaml) as fh:
        data = yaml.safe_load(fh)
    names = data["names"]
    if isinstance(names, dict):  # {0: 'a', 1: 'b'} form
        return [names[i] for i in sorted(names)]
    return list(names)


def resolve_dataset(label, local_data, project_env, version_env):
    """Return a path to data.yaml, preferring a local dataset over Roboflow.

    A local path avoids needing an API key at all, and avoids the version
    mismatch between what .env requests and what is checked out on disk.
    """
    local = local_data or os.getenv("DATA_YAML")
    if local:
        path = Path(local)
        if not path.is_absolute():
            path = HERE / path
        if not path.exists():
            sys.exit(f"Dataset not found: {path}")
        print(f"Using local {label} dataset: {path}")
        return path

    from roboflow import Roboflow

    print(f"Downloading {label} dataset from Roboflow...")
    try:
        rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
        project = rf.workspace(os.getenv("ROBOFLOW_WORKSPACE")).project(os.getenv(project_env))
        dataset = project.version(int(os.getenv(version_env, 1))).download("yolov8")
    except Exception as exc:
        sys.exit(f"Error downloading from Roboflow: {exc}\nCheck your API key and project details.")

    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        sys.exit(f"data.yaml not found in downloaded dataset at {dataset.location}")
    return data_yaml


def resolve_weights(name):
    """A local file if we have one, otherwise a name for Ultralytics to fetch."""
    path = Path(name)
    if not path.is_absolute():
        local = HERE / name
        if local.exists():
            return str(local)
    if path.exists():
        return str(path)
    print(f"'{name}' is not on disk; Ultralytics will download it.")
    return name


def verify_export_layout(onnx_path, num_classes):
    """The C++ decoder assumes YOLOv8's [1, 4+nc, N] output with no objectness
    column and no built-in NMS. yolo11 matches it; other families may not, and a
    mismatch would decode as confident nonsense rather than fail. Same check as
    the C++ side's construction-time assertion, run before the file is shipped."""
    try:
        import onnx
    except ImportError:
        print("onnx not installed; skipping output-layout check.")
        return
    model = onnx.load(str(onnx_path))
    dims = [d.dim_value or d.dim_param for d in
            model.graph.output[0].type.tensor_type.shape.dim]
    if len(dims) != 3 or dims[1] != 4 + num_classes:
        sys.exit(
            f"Refusing to export: unexpected output shape {dims}.\n"
            f"  expected [1, {4 + num_classes}, N] (4 box coords + {num_classes} classes).\n"
            f"  The C++ Detector cannot decode this layout."
        )
    print(f"Output layout OK: {dims}")


def train_and_export(label, run_name, output_name, project_env, version_env, argv=None):
    parser = argparse.ArgumentParser(description=f"Train and export the {label} model.")
    parser.add_argument("--data", help="path to a local data.yaml (skips Roboflow)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch", type=int, default=BATCH,
                        help="images per batch; lower this if training hits CUDA OOM")
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu")
    parser.add_argument("--weights", default=BASE_WEIGHTS,
                        help=f"starting weights (default: {BASE_WEIGHTS}); "
                             "e.g. yolo11s.pt")
    parser.add_argument("--name", default=run_name, help="run directory name under runs/detect")
    args = parser.parse_args(argv)

    data_yaml = resolve_dataset(label, args.data, project_env, version_env)
    wanted = expected_classes(data_yaml)
    print(f"{label} dataset: {len(wanted)} classes -> {wanted}")

    weights = resolve_weights(args.weights)
    model = YOLO(weights)
    print(f"Starting {label} model training from {args.weights}...")
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        patience=args.patience,
        batch=args.batch,
        device=args.device,
        project=str(HERE / "runs" / "detect"),
        name=args.name,
    )

    # Ultralytics auto-increments the run directory, so the weights must come
    # from the run that just finished -- never from a hardcoded 'train/'.
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        sys.exit(f"Training produced no weights at {best}")
    print(f"Best weights: {best}")

    trained = YOLO(str(best))
    got = [trained.names[i] for i in sorted(trained.names)]
    if got != wanted:
        sys.exit(
            f"Refusing to export: trained model reports the wrong classes.\n"
            f"  expected ({len(wanted)}): {wanted}\n"
            f"  got      ({len(got)}): {got}"
        )

    print(f"Exporting {label} model to ONNX...")
    onnx_path = trained.export(format="onnx", opset=12, simplify=True, dynamic=False)

    verify_export_layout(onnx_path, len(wanted))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / output_name
    shutil.copy(onnx_path, dest)
    print(f"\nExport complete.\nONNX model: {onnx_path}\nCopied to:  {dest}")
    return dest
