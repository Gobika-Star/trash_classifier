# scripts/classify_trash.py
from pathlib import Path
import shutil, os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array # pyright: ignore[reportMissingImports]
try:
    # tkinter for file dialogs
    from tkinter import Tk, filedialog
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

# OpenCV for camera capture
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

def get_paths():
    project_root = Path(__file__).resolve().parents[1]
    app_dir = project_root / "app"
    model_path = app_dir / "model.h5"
    output_dir = project_root / "sorted_trash"
    return project_root, model_path, output_dir

CLASS_NAMES = ["Biodegradable", "Non-biodegradable"]

def load_model_safe(model_path):
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}. Please run create_dummy_model.py or train_model.py first.")
        sys.exit(1)
    try:
        model = tf.keras.models.load_model(str(model_path))
        print("✅ Model loaded:", model_path)
        return model
    except Exception as e:
        print("❌ Failed to load model:", e)
        sys.exit(1)

def preprocess_image(img_path):
    x = load_img(str(img_path), target_size=(224,224))
    x = img_to_array(x) / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def classify_and_sort(model, img_path, output_dir):
    try:
        x = preprocess_image(img_path)
    except Exception as e:
        print(f"❌ Failed to load/preprocess {img_path}: {e}")
        return False
    preds = model.predict(x)
    class_index = int(np.argmax(preds, axis=1)[0])
    if class_index < 0 or class_index >= len(CLASS_NAMES):
        print("❌ Unexpected class index:", class_index)
        return False
    predicted_class = CLASS_NAMES[class_index]
    dest_dir = Path(output_dir) / predicted_class
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy(str(img_path), str(dest_dir))
        print(f"✅ {Path(img_path).name} → {predicted_class}")
        return True
    except Exception as e:
        print(f"❌ Could not copy file {img_path} to {dest_dir}: {e}")
        return False

def option_upload(model, output_dir):
    if not TK_AVAILABLE:
        print("❌ tkinter not available on this Python installation.")
        return
    Tk().withdraw()
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files","*.jpg *.jpeg *.png *.bmp")])
    if file_path:
        classify_and_sort(model, Path(file_path), output_dir)
    else:
        print("❌ No file selected.")

def option_camera(model, output_dir):
    if not CV2_AVAILABLE:
        print("❌ OpenCV (cv2) not installed; camera capture not available.")
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera. Check camera/driver.")
        return
    print("Camera opened. Press SPACE to capture, ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from camera.")
            break
        cv2.imshow("Camera - press SPACE to capture", frame)
        key = cv2.waitKey(1)
        if key % 256 == 27:  # ESC
            print("Exiting camera.")
            break
        elif key % 256 == 32:  # SPACE
            img_name = "captured_image.jpg"
            cv2.imwrite(img_name, frame)
            print(f"📸 Saved {img_name}")
            classify_and_sort(model, Path(img_name), output_dir)
            break
    cap.release()
    cv2.destroyAllWindows()

def option_multi(model, output_dir):
    if not TK_AVAILABLE:
        print("❌ tkinter not available on this Python installation.")
        return
    Tk().withdraw()
    file_paths = filedialog.askopenfilenames(title="Select images", filetypes=[("Image files","*.jpg *.jpeg *.png *.bmp")])
    if not file_paths:
        print("❌ No files selected.")
        return
    for fp in file_paths:
        classify_and_sort(model, Path(fp), output_dir)

def main():
    project_root, model_path, output_dir = get_paths()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_safe(model_path)

    print("\nChoose input method:")
    print("1 - Upload image from computer (file dialog)")
    print("2 - Take photo from camera (OpenCV)")
    print("3 - Select multiple images (file dialog)")

    choice = input("Enter choice (1/2/3): ").strip()
    if choice == "1":
        option_upload(model, output_dir)
    elif choice == "2":
        option_camera(model, output_dir)
    elif choice == "3":
        option_multi(model, output_dir)
    else:
        print("❌ Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
