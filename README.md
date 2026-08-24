# Isaac Assistant

The Isaac Assistant is a C++/Java application designed to enhance the gameplay experience of "The Binding of Isaac".

## Overview

Two versions of the assistant have been developed to explore different approaches and address incompatibilities found with YOLO models:

*   **Java Version**: Utilizes ModelZoo for object detection.
*   **C++ Version**: Leverages OpenCV and ONNX Runtime for real-time image processing.

The dual-development approach was meant to address some incompatibilities found with YOLO models and detections.
Despite some issues are attenuated by the C++ version, some other issues arose with the implementation.

## Key Features

*   **Custom Datasets & Models**: Two unique datasets were created and labeled using Roboflow: one for identifying the player's current items and another for recognizing available items on the floor. Custom YOLO models were trained on these datasets to ensure high detection accuracy.
*   **Real-time Analysis**: The assistant functions by capturing the game screen and processing the images to detect and identify items using the trained models.
*   **Strategic Suggestions**: Afterwards, it computes optimal item combinations and recipes using a hashing algorithm. It provides real-time suggestions to the player, sorting items by their rarity and effectiveness.

## Project Status

The **C++ version runs end to end**: it captures the screen, detects the crafting bag and the floor
pickups, and prints ranked recipe suggestions (`recipe_suggestor_cpp/`, `ctest` green). Two caveats:

*   Item quality and item pools are now **real game data**, extracted from the installed game into
    `collectibles.json`, so a suggested item is at least of the right quality from the right pool.
    Which item comes out is still an approximation: the selection RNG is not the Repentance one, so
    it stays opt-in behind `--heuristic` and labelled `(approx)`. Only the 19 recipes in
    `fixed.json` are guaranteed exact.
*   The floor detector is trained on the 23-class v2 dataset (mAP50 0.740). That figure covers
    only the 14 classes with validation data; `eternal_heart` in particular scores AP 0 and does
    not work. Read per-class AP in `models_training/README.md` rather than the headline number.

The **Java version is unfinished** and is kept only as the reference implementation of the crafting
hash (`crafting_computation/CraftingComputator.java`).

See `recipe_suggestor_cpp/TODO.md` for the remaining work.