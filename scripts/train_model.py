# scripts/train_model.py
import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # pyright: ignore[reportMissingImports]
from tensorflow.keras.applications import MobileNetV2 # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Model # pyright: ignore[reportMissingImports]
from tensorflow.keras.optimizers import Adam # pyright: ignore[reportMissingImports]
import numpy as np

# Define classes
classes = ['Biodegradable', 'Non-Biodegradable']
num_classes = len(classes)

# Dataset paths
train_dir = 'dataset/train'
test_dir = 'dataset/test'

def prepare_dataset():
    # Check existing folders
    existing_classes = os.listdir(train_dir)
    print('Existing train classes:', existing_classes)

    # Create binary folders
    bio_dir_train = os.path.join(train_dir, 'Biodegradable')
    non_bio_dir_train = os.path.join(train_dir, 'Non-Biodegradable')
    bio_dir_test = os.path.join(test_dir, 'Biodegradable')
    non_bio_dir_test = os.path.join(test_dir, 'Non-Biodegradable')

    os.makedirs(bio_dir_train, exist_ok=True)
    os.makedirs(non_bio_dir_train, exist_ok=True)
    os.makedirs(bio_dir_test, exist_ok=True)
    os.makedirs(non_bio_dir_test, exist_ok=True)

    # Map existing classes: Organic -> Biodegradable, others -> Non-Biodegradable
    biodegradable = ['Organic']
    for cls in existing_classes:
        cls_dir_train = os.path.join(train_dir, cls)
        cls_dir_test = os.path.join(test_dir, cls)
        if os.path.isdir(cls_dir_train):
            target_train = bio_dir_train if cls in biodegradable else non_bio_dir_train
            for img in os.listdir(cls_dir_train):
                shutil.move(os.path.join(cls_dir_train, img), os.path.join(target_train, f"{cls}_{img}"))
            os.rmdir(cls_dir_train)
        if os.path.isdir(cls_dir_test):
            target_test = bio_dir_test if cls in biodegradable else non_bio_dir_test
            for img in os.listdir(cls_dir_test):
                shutil.move(os.path.join(cls_dir_test, img), os.path.join(target_test, f"{cls}_{img}"))
            os.rmdir(cls_dir_test)

    print('Updated train classes:', os.listdir(train_dir))

def main():
    prepare_dataset()

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        classes=classes,
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        classes=classes,
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        classes=classes,
        shuffle=False
    )

    if train_generator.samples == 0:
        print("⚠ No images found in dataset. Creating a dummy multi-class model instead.")
        # Create dummy model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.save('model/trash_model.h5')
        print('Dummy model saved to model/trash_model.h5')
        return

    # Build model with transfer learning
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze base layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    epochs = 10  # Increase for better accuracy
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // train_generator.batch_size),
        validation_data=validation_generator,
        validation_steps=max(1, validation_generator.samples // validation_generator.batch_size),
        epochs=epochs
    )

    # Evaluate on test set
    if test_generator.samples > 0:
        test_loss, test_acc = model.evaluate(test_generator, steps=max(1, test_generator.samples // test_generator.batch_size))
        print(f'Test accuracy: {test_acc:.2f}')
    else:
        print("No test images, skipping evaluation.")

    # Save the model
    model.save('model/trash_model.h5')
    print('Model saved to model/trash_model.h5')

if __name__ == "__main__":
    main()
