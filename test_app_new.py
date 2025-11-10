from app.app import app
print('App created')
with app.test_client() as client:
    response = client.get('/')
    print('Status:', response.status_code)
    print('Data length:', len(response.get_data()))
    if len(response.get_data()) > 0:
        print('Data preview:', response.get_data()[:200].decode('utf-8'))
    else:
        print('No data')
