from app.app import app # pyright: ignore[reportMissingImports]
with app.test_client() as client:
    response = client.get('/')
    print('Status:', response.status_code)
    print('Data length:', len(response.get_data()))
    print('Data preview:', response.get_data()[:200].decode('utf-8'))
