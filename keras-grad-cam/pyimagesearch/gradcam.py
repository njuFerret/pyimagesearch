# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2


# -----------------------------
#   GRAD CAM CLASS
# -----------------------------
class GradCAM:
    def __init__(self, model, class_idx, layer_name=None):
        # Store the model, the class index used to measure the class activation map and the layer to be used when
        # visualizing the class activation map
        self.model = model
        self.class_idx = class_idx
        self.layer_name = layer_name
        # If the layer name is None, attempt to automatically find the target output layer
        if self.layer_name is None:
            self.layer_name = self.find_target_layer()

    def find_target_layer(self):
        # Attempt to find the final convolutional layer in the network by looping over the layers of the network in
        # the reverse order
        for layer in reversed(self.model.layers):
            # Check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # Otherwise, send error message saying that 4D output layer could not be found so the GradCAM algorithm could
        # be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heat_map(self, image, eps=1e-8):
        # Construct the gradient model by supplying:
        # (1) The inputs of the pre-trained model
        # (2) The output (presumably) final 4D layer in the network
        # (3) The output of the softmax activations from the model
        grad_model = Model(inputs=[self.model.inputs],
                           outputs=[self.model.get_layer(self.layer_name).output, self.model.output])
        # Record the operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Cast the image tensor to a float-32 data type, pass the image through the gradient model,
            # and grab the loss associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (conv_outputs, predictions) = grad_model(inputs)
            loss = predictions[:, self.class_idx]
        # Use the automatic differentiation to compute the gradients
        grads = tape.gradient(loss, conv_outputs)
        # Compute the guided gradients
        cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
        cast_grads = tf.cast(grads > 0, "float32")
        guided_grads = cast_conv_outputs * cast_grads * grads
        # The convolution and guided gradients have a batch dimension so let's grab the volume itself and
        # discard the batch
        conv_outputs = conv_outputs[0]
        guided_grads = guided_grads[0]
        # Compute the average of the gradient values, and using them as weights, compute the ponderation of the filters
        # with respect to the weights
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
        # Grab the spatial dimensions of the input image and resize the output class activation map
        # to match the input image dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # Normalize the heatmap such that all values lie in the range [0, 1], scale the resulting values
        # to the range [0, 255], and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # Return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS):
        # Apply the supplied color map to the heatmap and then overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1-alpha, 0)
        # Return the 2-tuple of the color mapped heatmap and the output overlaid image
        return heatmap, output