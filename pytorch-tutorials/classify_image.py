# -----------------------------
#   USAGE
# -----------------------------
# python classify_image.py --image data/images/boat.png

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch import config
from torchvision import models
import numpy as np
import argparse
import torch
import cv2


# -----------------------------
#   FUNCTIONS
# -----------------------------
def preprocess_image(image):
    # Swap the color channels from BGR to RGB, resize it and scale the pixel values to [0, 1] range
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    image = image.astype("float32") / 255.0
    # Subtract the ImageNet mean, divide by the ImageNet standard deviation,
    # set "channels first" ordering and add a batch dimension
    image -= config.MEAN
    image /= config.STD
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    # Return the preprocessed image
    return image


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
ap.add_argument("-m", "--model", type=str, default="vgg16",
                choices=["vgg16", "vgg19", "inception", "densenet", "resnet"],
                help="Name of the pre-trained network to use")
args = vars(ap.parse_args())

# Define a dictionary that maps model names to their classes inside torchvision
MODELS = {"vgg16": models.vgg16(pretrained=True), "vgg19": models.vgg19(pretrained=True),
          "inception": models.inception_v3(pretrained=True), "densenet": models.densenet121(pretrained=True),
          "resnet": models.resnet50(pretrained=True)}

# Load the network weights file from disk, flash it to the current device and set it to evaluation mode
print("[INFO] Loading {}...".format(args["model"]))
model = MODELS[args["model"]].to(config.DEVICE)
model.eval()

# Load the image from disk, clone it (in order to draw on it later) and preprocess the image
print("[INFO] Loading image...")
image = cv2.imread(args["image"])
orig = image.copy()
image = preprocess_image(image)

# Convert the preprocessed image to a torch tensor and flash it to the current device
image = torch.from_numpy(image)
image = image.to(config.DEVICE)

# Load the preprocessed the ImageNet labels
print("[INFO] Loading ImageNet labels...")
imagenetLabels = dict(enumerate(open(config.IN_LABELS)))

# Classify the image and extract the predictions
print("[INFO] Classifying image with '{}'...".format(args["model"]))
logits = model(image)
probabilities = torch.nn.Softmax(dim=-1)(logits)
sortedProba = torch.argsort(probabilities, dim=-1, descending=True)

# Loop over the predictions and display the rank-5 predictions and corresponding probabilities to the terminal
for (i, idx) in enumerate(sortedProba[0, :5]):
    print("{}. {}: {:.2f}%".format(i, imagenetLabels[idx.item()].strip(), probabilities[0, idx.item()] * 100))

# Draw the top prediction on the image and display the image to the screen
(label, prob) = (imagenetLabels[probabilities.argmax().item()], probabilities.max().item())
cv2.putText(orig, "Label: {}, {:.2f}%".format(label.strip(), prob * 100), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)


