import tensorflow as tf
import numpy as np
import cv2

# Function to generate a Grad-CAM heatmap for a given image and model
def make_gradcam_heatmap(img_array, model):
    # Get the last convolutional layer of the model by name
    # This layer is used because it retains spatial information
    last_conv_layer = model.get_layer("last_conv")

    # Create a model that outputs both:
    # 1. The activations of the last convolutional layer
    # 2. The final model predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # Forward pass through the model
        conv_outputs, predictions = grad_model(img_array)

        # Use the predicted class score as the loss
        # Index 0 is used because this is a binary classification model
        loss = predictions[:, 0]

    # Compute gradients of the loss with respect to the convolutional outputs
    grads = tape.gradient(loss, conv_outputs)

    # Average the gradients over the spatial dimensions (height and width)
    # This produces one importance weight per feature map
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Remove the batch dimension from the convolutional outputs
    conv_outputs = conv_outputs[0]

    # Compute the weighted sum of the feature maps
    # This highlights the regions most important for the prediction
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to keep only positive values
    heatmap = tf.maximum(heatmap, 0)

    # Normalize the heatmap to values between 0 and 1
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    # Convert the heatmap to a NumPy array and return it
    return heatmap.numpy()


# Function to overlay the Grad-CAM heatmap on the original image
def overlay_heatmap(img_path, heatmap, output_path, alpha=0.4):
    # Load the original image using OpenCV
    img = cv2.imread(img_path)

    # Resize the image to match the model input size
    img = cv2.resize(img, (150, 150))

    # Resize the heatmap to match the image size
    heatmap = cv2.resize(heatmap, (150, 150))

    # Convert the heatmap values to 8-bit format
    heatmap = np.uint8(255 * heatmap)

    # Apply a color map to the heatmap for better visualization
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay the heatmap on the original image
    # Alpha controls the transparency of the heatmap
    superimposed = heatmap * alpha + img

    # Save the resulting image to disk
    cv2.imwrite(output_path, superimposed)
