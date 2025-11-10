from app.app import predict_trash # pyright: ignore[reportMissingImports]
import os

test_img = 'app/static/uploads/captured.jpg'
if os.path.exists(test_img):
    result = predict_trash(test_img)
    print('Test prediction on uploaded image:', result)
else:
    print('Test image not found')
    print('Available images:', os.listdir('app/static/uploads') if os.path.exists('app/static/uploads') else 'No dir')
