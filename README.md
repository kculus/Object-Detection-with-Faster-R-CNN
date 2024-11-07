# Object-Detection-with-Faster-R-CNN
This repository contains an object detection pipeline using a Faster R-CNN model fine-tuned for custom object detection tasks. The project includes scripts for model training, inference on test images, and a helper file that maps image file names to their corresponding IDs.

#Requirements
To run the project, ensure you have the following libraries installed:
torch
torchvision
Pillow
You can install them using pip:
pip install torch torchvision Pillow

#Data Augmentation
Data augmentation is used to artificially increase the diversity of the training dataset by applying random transformations to the images. This helps in cases where the dataset may have limited examples or class imbalance.

Why Data Augmentation?
Improves Generalization: By showing the model various transformations of the same data, data augmentation helps the model generalize better and not memorize specific features.
Reduces Overfitting: Data augmentation prevents overfitting by adding variability, allowing the model to learn more robust features rather than memorizing specific instances.
Balances Rare Classes: In this project, some classes (e.g., futbol_sahası and silo) have fewer examples. Data augmentation helps improve the model's ability to recognize these underrepresented classes.
Increases Model Robustness: Transformations like flipping, rotation, and color adjustments make the model more robust to changes in perspective, orientation, and lighting.
Augmentation Techniques Used
In this project, we apply the following data augmentation techniques to the rare classes:

RandomHorizontalFlip: Flips the images horizontally with a certain probability, which helps the model recognize objects from different orientations.
RandomRotation: Randomly rotates images by a specified degree range, adding variation in object appearance.
ColorJitter: Adjusts brightness, contrast, saturation, and hue to simulate different lighting conditions.
These augmentations are applied only to rare classes to enhance the model's performance on them.

#Training the Model
To train the Faster R-CNN model, use the object_detection.py script. This script:

Loads a custom dataset in COCO format.
Applies data augmentation techniques (e.g., random flipping and color jittering).
Trains a Faster R-CNN model with a ResNet-50 backbone.

Run the Training Script
python object_detection.py

This script will:

Load and preprocess the dataset.
Apply data augmentation techniques.
Train the model for the specified number of epochs.
Save the trained model weights to fasterrcnn_finetuned2.pth.
After training, you can find the model weights in the current directory.

#Inference
The infer.py script is used to perform inference on new images using the trained model. This script loads the model weights from fasterrcnn_finetuned2.pth and makes predictions on the images located in the test-images folder.

This will:

Load the trained Faster R-CNN model from fasterrcnn_finetuned2.pth.
Perform object detection on each image in the test-images folder.
Use the image_file_name_to_image_id.json file to map image file names to unique image IDs.
Save the inference results to inference_results2.json.
The output JSON file (inference_results2.json) contains the following information for each detected object:

image_id: The unique ID of the image (retrieved from image_file_name_to_image_id.json).
category_id: The predicted class ID of the detected object.
bbox: Bounding box coordinates in [x, y, width, height] format.
score: Confidence score for the detected object.

#About image_file_name_to_image_id.json
The image_file_name_to_image_id.json file is a helper JSON file that maps each image file name (e.g., image1.jpg) to a unique image ID (e.g., 123). This mapping is crucial for cases where the dataset or evaluation framework requires each image to have a unique identifier. During inference, the script uses this file to assign an image ID to each detection result, which is then stored in inference_results2.json. If an image file name is not found in this mapping file, the script skips that image and outputs a warning message.

#Notes
Ensure that your dataset is in COCO format and correctly structured.
Customize the data augmentation techniques in object_detection.py to better suit your data requirements.
You can adjust the test-images folder path in infer.py if your test images are in a different directory.
