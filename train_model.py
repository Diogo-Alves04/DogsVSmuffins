# Import ImageDataGenerator to load images from directories,
# apply preprocessing, and generate batches for training and validation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import layers used to build the Convolutional Neural Network (CNN)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

# Import Model to define the model using the Functional API
from tensorflow.keras.models import Model

# Import Adam optimizer to train the model
from tensorflow.keras.optimizers import Adam

# Import os to handle directory creation and file saving
import os


# Create an ImageDataGenerator for preprocessing images
# rescale=1./255 normalizes pixel values from [0,255] to [0,1]
# validation_split=0.2 reserves 20% of data for validation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Create the training data generator
# Images are loaded from the "dataset" directory
# target_size resizes images to 150x150
# batch_size defines how many images are processed at once
# class_mode="binary" is used for binary classification
# subset="training" selects the training portion of the data
train_gen = datagen.flow_from_directory(
    "dataset",
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

# Create the validation data generator
# Uses the same dataset but selects the validation subset
val_gen = datagen.flow_from_directory(
    "dataset",
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Print the mapping between class names and numeric labels
# Example: dog -> 0 | muffin -> 1
print("Class indices:", train_gen.class_indices)


# Define the input layer of the model
# Input images have shape 150x150 with 3 color channels (RGB)
inputs = Input(shape=(150, 150, 3))

# First convolutional layer
# 32 filters, 3x3 kernel, ReLU activation
x = Conv2D(32, (3, 3), activation="relu", name="conv_1")(inputs)

# Max pooling layer to reduce spatial dimensions
x = MaxPooling2D(2, 2)(x)

# Second convolutional layer
# 64 filters to learn more complex features
x = Conv2D(64, (3, 3), activation="relu", name="conv_2")(x)

# Max pooling layer
x = MaxPooling2D(2, 2)(x)

# Third convolutional layer
# 128 filters for even deeper feature extraction
x = Conv2D(128, (3, 3), activation="relu", name="last_conv")(x)

# Max pooling layer
x = MaxPooling2D(2, 2)(x)

# Flatten the 3D feature maps into a 1D vector
x = Flatten()(x)

# Fully connected (dense) layer with 512 neurons
x = Dense(512, activation="relu")(x)

# Dropout layer to reduce overfitting by randomly disabling 50% of neurons
x = Dropout(0.5)(x)

# Output layer
# Single neuron with sigmoid activation for binary classification
outputs = Dense(1, activation="sigmoid")(x)

# Create the model by specifying inputs and outputs
model = Model(inputs, outputs)

# Compile the model
# Adam optimizer with a low learning rate
# Binary crossentropy loss for binary classification
# Accuracy as evaluation metric
model.compile(
    optimizer=Adam(0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the model
# Uses the training generator and validates on the validation generator
# Runs for 10 epochs
model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

# Create a directory called "model" if it does not exist
os.makedirs("model", exist_ok=True)

# Save the trained model to a file
model.save("model/dogs_vs_muffins_cnn.h5")
