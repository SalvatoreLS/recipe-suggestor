"""Train and export the Bag of Crafting detector."""

from train_common import train_and_export

if __name__ == "__main__":
    train_and_export(
        label="BoC",
        run_name="boc",
        output_name="boc_best.onnx",
        project_env="ROBOFLOW_PROJECT_BOC",
        version_env="ROBOFLOW_VERSION_BOC",
    )
