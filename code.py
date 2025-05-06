// Language : python

import os
import random
import shutil

splitsize = .85
categories = []

source_folder = r"C:\\Users\\siyas\\Desktop\\projects\\sharks"
folders = os.listdir(source_folder)
print(folders)

for subfolder in folders:
    if os.path.isdir(source_folder + "/" + subfolder):
        categories.append(subfolder)

categories.sort()
print(categories)

# create a target folder
target_folder = r"C:\\Users\\siyas\\Desktop\\projects\\sharks"
existDataSetPath = os.path.exists(target_folder)
if not existDataSetPath:
    os.mkdir(target_folder)

# Path to the original dataset
source_folder = r"C:/Users/siyas/Desktop/projects/sharks"
# Path where split dataset will be stored
target_folder = r"C:/Users/siyas/Desktop/projects/sharks/dataset_for_model"
# Ensure the target_folder exists
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# Create base train and validate folders
trainPath = os.path.join(target_folder, "train")
validatePath = os.path.join(target_folder, "validate")

os.makedirs(trainPath, exist_ok=True)
os.makedirs(validatePath, exist_ok=True)

# Train-validation split ratio
splitsize = 0.85

# List of shark categories (subfolder names)
categories = ['basking', 'blacktip', 'blue', 'bull', 'hammerhead', 'lemon',
              'mako', 'nurse', 'sand tiger', 'thresher', 'tiger', 'whale',
              'white', 'whitetip']

# Function to split the data
def split_data(SOURCE, TRAINING, VALIDATION, SPLIT_SIZE):
    files = []

    for filename in os.listdir(SOURCE):
        file = SOURCE + "/" + filename
        if os.path.getsize(file) > 0 and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            files.append(filename)
        else:
            print(filename + " is 0 length or not an image, ignored...")

    print(f"{len(files)} valid images found in {SOURCE}")

    trainingLength = int(len(files) * SPLIT_SIZE)
    shuffleSet = random.sample(files, len(files))
    trainingSet = shuffleSet[:trainingLength]
    validationSet = shuffleSet[trainingLength:]

    for filename in trainingSet:
        shutil.copy(os.path.join(SOURCE, filename), os.path.join(TRAINING, filename))

    for filename in validationSet:
        shutil.copy(os.path.join(SOURCE, filename), os.path.join(VALIDATION, filename))

ValidGenerator = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=(0, 0.2)
).flow_from_directory(ValidPath, target_size=(320, 320), batch_size=32)

# Build the model using MobileNetV3 Large
baseModel = MobileNetV3Large(weights="imagenet", include_top=False)

x = baseModel.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='hard_swish')(x)
x = Dense(256, activation='hard_swish')(x)
x = Dense(128, activation='hard_swish')(x)

predictionLayer = Dense(trainGenerator.num_classes, activation='softmax')(x)
model = Model(inputs=baseModel.input, outputs=predictionLayer)

print(model.summary())

from PIL import Image
import os

def convert_images_to_rgb(folder):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        if img.mode in ("P", "RGBA"):
                            img = img.convert("RGB")
                            img.save(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Run it on both train and validation folders
convert_images_to_rgb(r"C:/Users/siyas/Desktop/projects/sharks/dataset_for_model/train")
convert_images_to_rgb(r"C:/Users/siyas/Desktop/projects/sharks/dataset_for_model/validate")

# Freeze the layers of the MobileNetV3 (already trained)
for layer in model.layers[:-5]:
    layer.trainable = False

# Compile
optimizer = Adam(learning_rate=0.0001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

# Train the model and store the training history
history = model.fit(trainGenerator, validation_data=ValidGenerator, epochs=30)

# Now you can print the final accuracies
print("Final Training Accuracy:", history.history['accuracy'][-1])

