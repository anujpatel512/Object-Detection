import tensorflow as tf
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load and prepare the dataset
dataset_path = '...'

def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    labels = []
    for obj in root.findall('object'):
        labels.append(obj.find('name').text)
        bndbox = obj.find('bndbox')
        boxes.append([
            int(bndbox.find('xmin').text),
            int(bndbox.find('ymin').text),
            int(bndbox.find('xmax').text),
            int(bndbox.find('ymax').text)
        ])
    return boxes, labels

images = []
annotations = []
image_paths = glob.glob(os.path.join(dataset_path, 'images', '*.jpg'))

for img_path in image_paths:
    xml_path = img_path.replace('images', 'annotations').replace('.jpg', '.xml')
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = preprocess_input(img)
    images.append(img)
    boxes, labels = parse_annotation(xml_path)
    annotations.append({'boxes': boxes, 'labels': labels})

images = np.array(images)

train_images, val_images, train_annotations, val_annotations = train_test_split(
    images, annotations, test_size=0.2, random_state=42)

# Load pre-trained model and set up transfer learning
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet')

base_model.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels,
                    validation_data=(val_images, val_labels),
                    epochs=20)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_images, val_labels)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%")
