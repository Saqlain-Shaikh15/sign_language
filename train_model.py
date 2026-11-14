import pickle
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# parse mode so one script trains either ISL or ASL model
parser = argparse.ArgumentParser(description='Train sign language model')
parser.add_argument('--mode', type=str, choices=['isl', 'asl'], required=True, help='Sign language mode: isl or asl') 
args = parser.parse_args()

# load correct dataset based on mode
with open(f'./data_{args.mode}.pickle', 'rb') as f:
    data_dict = pickle.load(f)

lengths = {len(s) for s in data_dict['data']}
assert len(lengths) == 1, f'Inconsistent sample lengths: {lengths}'

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f'{score * 100:.2f}% of samples were classified correctly!')
print(classification_report(y_test, y_predict))

# changed: save to mode-specific model file
model_path = f'model_{args.mode}.p'
with open(model_path, 'wb') as f:
    pickle.dump({'model': model}, f)

print(f'Model saved to {model_path}')