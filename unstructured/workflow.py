#%%
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import datetime, os
import matplotlib.pyplot as plt
from IPython.display import Image
from sklearn.model_selection import train_test_split

# Check GPU acceleration enabled
# print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

# Import data
labels_csv = pd.read_csv("data/labels.csv")
# Create images filepaths
filenames = ["data/train/" + fname + ".jpg" for fname in labels_csv["id"]]
# Convert labels to be in boolean format
labels = np.array(labels_csv["breed"])
unique_labels = np.unique(labels)
boolean_labels = [label == unique_labels for label in labels]

# Setup X and Y variables
x = filenames
y = boolean_labels

# Split data into train and validation
NUM_IMAGES = 1000
x_train, x_val, y_train, y_val = train_test_split(
    x[:NUM_IMAGES], y[:NUM_IMAGES], test_size=0.2, random_state=42
)

# Global variables
IMG_SIZE = 224

# Preprocessing images (Convert to tensors)
def process_image(image_path, IMG_SIZE=IMG_SIZE):
    # Read in an image file
    image = tf.io.read_file(image_path)
    # Turn the jpeg image into numerical Tensor with RGB channels
    image = tf.image.decode_jpeg(image, channels=3)
    # Convert color channel values from 0-255 to 0-1 values
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize the image
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image


def get_image_label(image_path, label):
    image = process_image(image_path)
    return image, label


# Creates batches of data out of image (X) and label (Y) pairs
# Training = Shuffle data
# Validation = Dont shuffle data
# Test = Accepts as input, no labels
def create_data_batches(x, y=None, valid_data=False, test_data=False):
    BATCH_SIZE = 32
    if test_data:
        print("Creating test data batches")
        # Create dataset out of tensors
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))
        # Image processing and convert to batch
        data_batch = data.map(process_image).batch(BATCH_SIZE)
        return data_batch
    elif valid_data:
        print("Creating validation data batches")
        # Create dataset out of tensors
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y)))
        # Image processing and convert to batch
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
        return data_batch
    else:
        print("Create training data batches")
        # Create dataset out of tensors
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y)))
        # Shuffling filenames and labels BEFORE processing image for efficiency
        data = data.shuffle(buffer_size=len(x))
        # Image processing and convert to batch
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
        return data_batch


# Create training and validation batches
train_data = create_data_batches(x_train, y_train)
val_data = create_data_batches(x_val, y_val, valid_data=True)

# Setup input shape to the model (Batch, height, width, color channels)
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3]
# Setup output shape of our model
OUTPUT_SHAPE = len(unique_labels)
# Setup model URL from TensorFlow Hub
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5"

# Create a function which builds a keras model
def create_model():
    # Setup the model layers
    model = tf.keras.Sequential(
        [
            hub.KerasLayer(MODEL_URL),
            tf.keras.layers.Dense(units=OUTPUT_SHAPE, activation="softmax"),
        ]
    )
    # Compile the model
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    # Build the model
    model.build(INPUT_SHAPE)
    return model


# Callback to track logs whenever we run experiment
def create_tensorboard_callback():
    logdir = os.path.join("./logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return tf.keras.callbacks.TensorBoard(logdir)


# Callback for early stopping (Prevent overfitting)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3)

# Train model
def train_model():
    NUM_EPOCHS = 100
    # Create model
    model = create_model()
    # Create new TensorBoard session everytime we train a model
    tensorboard = create_tensorboard_callback()
    # Fit the model to the data passing it the callbacks we created
    model.fit(
        x=train_data,
        epochs=NUM_EPOCHS,
        validation_data=val_data,
        validation_freq=1,
        callbacks=[early_stopping],
    )
    # Return the fitted model
    return model


# Convert probabilities into their respective labels
def get_pred_label(prediction_probabilities):
    return unique_labels[np.argmax(prediction_probabilities)]


# Unbatch dataset
def unbatchify(data):
    images = []
    labels = []
    for image, label in data.unbatch().as_numpy_iterator():
        images.append(image)
        labels.append(get_pred_label(label))
    return images, labels


# Comparison between predicted to truth label
def plot_pred(prediction_probabilities, labels, images, n=1):
    pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]
    # Get the predicted label
    pred_label = get_pred_label(pred_prob)
    # Plot image and remove ticks
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    # Comparison
    plt.title(f"{pred_label} {np.max(pred_prob)*100:2.0f}% {true_label}")


# Save model
def save_model(model, suffix=None):
    # Create directory with pathname with current time
    modeldir = os.path.join(
        "./models", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    # Format of model
    model_path = f"{modeldir}-{suffix}.h5"
    print(f"Saved model to: {model_path}")
    model.save(model_path)
    return model_path


# Load model
def load_model(model_path):
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(
        model_path, custom_objects={"KerasLayer": hub.KerasLayer}
    )
    return model


# Train model
model = train_model()
# Save model
save_model(model, suffix="100-image-set-mobilenetv2-Adam")
# Load model
model = load_model("models/20210512-224432-100-image-set-mobilenetv2-Adam.h5")
# Predict
predictions = model.predict(val_data, verbose=1)
# Unbatch
val_images, val_labels = unbatchify(val_data)
# Compare
plot_pred(predictions, val_labels, val_images, 1)
