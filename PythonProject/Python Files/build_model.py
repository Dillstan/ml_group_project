##MIX of GPT AND OWN CODE

from scipy.io import loadmat
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from tensorflow.keras.callbacks import ModelCheckpoint, Callback  # Import necessary callbacks

TEST_NUM = 2
BATCH_SIZE = 128
IMG_SIZE = (160, 160)

root='../../../ml_project_files/Models/mini_test_'+str(TEST_NUM)
epoch_dir=root+"/epoch"
log_dir=root+'/logs'


# Make a directory for saving training logs if it doesn't exist
os.makedirs(log_dir, exist_ok=True)
os.makedirs(epoch_dir, exist_ok=True)


# Save model and log training history after each epoch
checkpoint_path_h5 = epoch_dir+"/my_model_epoch_{epoch:02d}.h5"  # Save model with epoch number in the filename
checkpoint_path_keras = epoch_dir+"/my_model_epoch_{epoch:02d}.keras"  # Save model with epoch number in the filename

# Create ModelCheckpoint callback
checkpoint_callback_h5 = ModelCheckpoint(
    filepath=checkpoint_path_h5,
    save_weights_only=False,
    save_best_only=False,
    monitor='loss',
    mode='min',
    verbose=1
)

checkpoint_callback_keras = ModelCheckpoint(
    filepath=checkpoint_path_keras,
    save_weights_only=False,
    save_best_only=False,
    monitor='loss',
    mode='min',
    verbose=1
)

class TrainingHistoryLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Create a new log file for each epoch
        log_filename = f"{log_dir}/epoch_{epoch + 1}.txt"  # Use epoch number for filename
        with open(log_filename, "w") as log_file:
            # Write loss and accuracy to the file
            log_file.write(f"Epoch: {epoch + 1}\n")  # Epoch number
            for key, value in logs.items():
                log_file.write(f"{key}: {value}\n")  # Log each metric

print("Loading imdb.mat...")
data = loadmat('../../../ml_project_files/imdb_crop/imdb.mat')
print("Keys in .mat file:", data.keys())

imdb = data['imdb']
print("imdb type:", type(imdb))
print("imdb shape:", imdb.shape)

# Extract raw fields
print("Extracting full_path and celeb_id...")
file_paths_raw = np.squeeze(imdb['full_path'])
celeb_ids_raw  = np.squeeze(imdb['celeb_id'])

print("file_paths_raw type:", type(file_paths_raw))
print("celeb_ids_raw type:", type(celeb_ids_raw))

# These are object arrays wrapped inside .item()
file_paths_raw = file_paths_raw.item()[0]
celeb_ids_raw  = celeb_ids_raw.item()[0]

num_imgs = len(file_paths_raw)
print("Number of images:", num_imgs)
print("Number of labels:", len(celeb_ids_raw))

print("First raw path entry:", file_paths_raw[0])
print("First raw celeb_id:", celeb_ids_raw[0])

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print("⚠️ cv2.imread failed for:", path)
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img / 255.0

# Build full paths
print("Building image paths...")
paths = ['imdb_crop/' + str(p[0]) for p in file_paths_raw]
labels = celeb_ids_raw.astype(int)

print("Sample constructed path:", paths[0])
print("Sample label:", labels[0])

print("Unique celeb_ids:", len(np.unique(labels)))

# TensorFlow dataset
print("Creating TensorFlow dataset...")
dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img, label

dataset = dataset.map(load_and_preprocess)
dataset = dataset.shuffle(20000)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.repeat()
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # For optimized performance


print("Dataset pipeline created.")

# Peek at one batch
for imgs, lbls in dataset.take(1):
    print("Batch image shape:", imgs.shape)
    print("Batch label shape:", lbls.shape)
    print("First label in batch:", lbls[0].numpy())

num_classes = len(np.unique(labels))
print("Number of classes (num_classes):", num_classes)

# Model
print("Building model...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled.")
model.summary()

#print avail gpus
print(tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

# Begin training
print("Starting training...")
print()

# Create an instance of the logger
history_logger = TrainingHistoryLogger()
history = model.fit(dataset, epochs=10, steps_per_epoch=num_imgs//BATCH_SIZE, callbacks=[checkpoint_callback_h5, checkpoint_callback_keras, history_logger])

print("Training finished.")

# Save the trained model
model.save(root+"/my_model_h5.h5")  # Save in HDF5 format
model.save(root+"/my_model_tf.keras")  # Save as a TensorFlow SavedModel





# ---- Prediction section (guarded, since img_tensor wasn't defined) ----
print("Preparing for prediction...")

# Example: reuse a batch image
img_tensor = imgs[0]
pred = model.predict(img_tensor[tf.newaxis, ...])
pred_id = np.argmax(pred)

print("Predicted celeb_id:", pred_id)

# Map id to name
celeb_names = np.squeeze(imdb['celeb_names']).item()
print("Total celeb names:", len(celeb_names))

if pred_id < len(celeb_names):
    print("Predicted Name:", celeb_names[pred_id][0])
else:
    print("⚠️ Predicted ID out of range for celeb_names")
