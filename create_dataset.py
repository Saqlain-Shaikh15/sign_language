import os
import pickle
import argparse # added
import mediapipe as mp
import cv2

# added: parse mode so one script handles both ISL and ASL datasets
parser = argparse.ArgumentParser(description='Create dataset for sign language') # added
parser.add_argument('--mode', type=str, choices=['isl', 'asl'], required=True, help='Sign language mode: isl or asl') # added
args = parser.parse_args() # added

# changed: use correct mp.solutions.hands (fixes broken import in create_dataset_asl.py which had mp.vision.hands)
mp_hands = mp.solutions.hands # changed
mp_drawing = mp.solutions.drawing_utils # changed
mp_drawing_styles = mp.solutions.drawing_styles # changed

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# changed: pick image dir and output file based on mode
DATA_DIR = f'./{args.mode.upper()}_images' # changed
output_file = f'data_{args.mode}.pickle' # added

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            hand_data = []
            for hand_landmarks in results.multi_hand_landmarks:
                single_hand_data = []
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                    single_hand_data.append(x)
                    single_hand_data.append(y)

                hand_data.append(single_hand_data)

            # Ensure consistent shape for one or two hands
            if len(hand_data) == 1:  # Only one hand detected
                hand_data.append([0] * len(hand_data[0]))  # Pad with zeros for the second hand

            # Flatten the hand data into a single list
            data_aux.extend(hand_data[0])
            data_aux.extend(hand_data[1])

            data.append(data_aux)
            labels.append(dir_)

# changed: write to mode-specific pickle file
f = open(output_file, 'wb') # changed
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

print(f'Dataset saved to {output_file}') # added