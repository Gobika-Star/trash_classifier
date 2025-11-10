# scripts/create_dummy_model.py
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # pyright: ignore[reportMissingImports]
import os

model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(),
    Flatten(),
    Dense(6, activation='softmax')  # 6 classes: Plastic, Paper, Glass, Metal, Organic, Other
])

# Create model folder if it does not exist
os.makedirs('app/model', exist_ok=True)

# Save the model
model.save('app/model/trash_model.h5')
print("Dummy model created at app/model/trash_model.h5")
