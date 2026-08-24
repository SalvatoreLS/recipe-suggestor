"""Train and export the floor-pickups detector."""

from train_common import train_and_export

if __name__ == "__main__":
    train_and_export(
        label="Floor",
        run_name="floor",
        output_name="floor_best.onnx",
        project_env="ROBOFLOW_PROJECT_FLOOR",
        version_env="ROBOFLOW_VERSION_FLOOR",
    )
