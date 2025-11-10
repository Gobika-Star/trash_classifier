import pandas as pd
import os

csv_path = 'app/data/predictions.csv'
print('CSV exists:', os.path.exists(csv_path))
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print('CSV shape:', df.shape)
    print('Columns:', list(df.columns))
    print('Sample data:', df.head().to_dict('records') if not df.empty else 'Empty')
else:
    print('CSV does not exist, creating empty one...')
    df = pd.DataFrame(columns=['filename', 'predicted_class', 'confidence', 'timestamp'])
    df.to_csv(csv_path, index=False)
    print('Empty CSV created.')
