from app.app import app # pyright: ignore[reportMissingImports]
import os

print('App imported')
print('Templates folder:', app.template_folder)
print('Static folder:', app.static_folder)
print('Upload folder:', app.config['UPLOAD_FOLDER'])

print('Templates dir exists:', os.path.exists(app.template_folder))
print('Static dir exists:', os.path.exists(app.static_folder))
print('Index template exists:', os.path.exists(os.path.join(app.template_folder, 'index.html')))
print('CSS exists:', os.path.exists(os.path.join(app.static_folder, 'css', 'style.css')))
print('Model exists:', os.path.exists('model/trash_model.h5'))
print('Data exists:', os.path.exists('data/predictions.csv'))
print('Uploads dir exists:', os.path.exists(app.config['UPLOAD_FOLDER']))
print('Result template exists:', os.path.exists(os.path.join(app.template_folder, 'result.html')))

with app.test_client() as client:
    response = client.get('/')
    print('Status:', response.status_code)
    print('Data length:', len(response.get_data()))
    if len(response.get_data()) > 0:
        print('Data preview:', response.get_data()[:200].decode('utf-8'))
    else:
        print('No data')
