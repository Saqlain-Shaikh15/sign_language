import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model_isl.p', 'rb'))
model = model_dict['model']

# Initialize VideoCapture
cap = cv2.VideoCapture(0)  # Change index if needed
if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Correct labels_dict for A-Z and 0-9
labels_dict = {
    **{i: chr(65 + i) for i in range(26)},  # Map 0-25 to A-Z
    **{i: str(i - 26) for i in range(26, 36)}  # Map 26-35 to 0-9
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to read frame from camera. Retrying...")
        continue

    print("Camera frame captured successfully.")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = []
    x_ = []
    y_ = []

    if results.multi_hand_landmarks:
        print("Hand landmarks detected.")
        hand_data_list = []  # To store data for each hand
        for hand_landmarks in results.multi_hand_landmarks:
            single_hand_data = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
                single_hand_data.append(x)
                single_hand_data.append(y)
            hand_data_list.append(single_hand_data)

        # Ensure input has data for both hands
        if len(hand_data_list) == 1:  # Only one hand detected
            print("Only one hand detected. Padding with zeros for the second hand.")
            hand_data_list.append([0] * 42)  # Pad with zeros for the missing hand

        # Flatten the hand data into a single list
        data_aux.extend(hand_data_list[0])  # Data for the first hand
        data_aux.extend(hand_data_list[1])  # Data for the second hand

        if len(data_aux) == 84:  # Ensure the input has the correct shape
            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                print("Predicted character:", predicted_character)

                # Draw bounding box and predicted character
                x1 = int(min(x_) * frame.shape[1]) - 10
                y1 = int(min(y_) * frame.shape[0]) - 10
                x2 = int(max(x_) * frame.shape[1]) + 10
                y2 = int(max(y_) * frame.shape[0]) + 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            except Exception as e:
                print("Error during prediction:", e)
        else:
            print("Error: Input data has incorrect shape.")
    else:
        print("No hands detected.")

    # Draw landmarks and connections for visualization
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # Image to draw on
                hand_landmarks,  # Detected hand landmarks
                mp_hands.HAND_CONNECTIONS,  # Draw connections between landmarks
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    cv2.imshow('Sign Language Detector', frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        print("Exit key pressed. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()