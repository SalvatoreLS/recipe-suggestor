# Isaac Assistant

The Isaac Assistant is a C++/Java application designed to enhance the gameplay experience of "The Binding of Isaac".

## Overview

Two versions of the assistant have been developed to explore different approaches and address incompatibilities found with YOLO models:

*   **Java Version**: Utilizes ModelZoo for object detection.
*   **C++ Version**: Leverages OpenCV and LibTorch for real-time image processing.

The dual-development approach was meant to address some incompatibilities found with YOLO models.
Despite some issues are attenuated by the C++ version, some other issues arose with the implementation.

## Key Features

*   **Custom Datasets & Models**: Two unique datasets were created and labeled using Roboflow: one for identifying the player's current items and another for recognizing available items on the floor. Custom YOLO models were trained on these datasets to ensure high detection accuracy.
*   **Real-time Analysis**: The assistant functions by capturing the game screen and processing the images to detect and identify items using the trained models.
*   **Strategic Suggestions**: Afterwards, it computes optimal item combinations and recipes using a hashing algorithm. It provides real-time suggestions to the player, sorting items by their rarity and effectiveness.

## Project Status

Due to model incompatibility and time the project is not completed in both versions and it is not working yet.