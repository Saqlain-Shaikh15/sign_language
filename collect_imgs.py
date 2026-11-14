
import os
import cv2
import argparse

def classLetterMapping(c):
    if c < 26:
        return chr(c + ord('A'))
    return str(c - 26)


parser = argparse.ArgumentParser(description='Collect sign language images')
parser.add_argument(
    '--mode',
    type=str,
    choices=['isl', 'asl'],
    required=True,
    help='Sign language mode: isl or asl'
)
args = parser.parse_args()

DATA_DIR = f'./{args.mode.upper()}_images'
os.makedirs(DATA_DIR, exist_ok=True)

number_of_classes = 36
dataset_size = 200

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    os.makedirs(class_dir, exist_ok=True)

    # Find already collected images
    images = [
        file for file in os.listdir(class_dir)
        if file.endswith('.jpg')
    ]

    # Skip if the class is already complete
    if len(images) >= dataset_size:
        print(f"Class {classLetterMapping(j)} already completed. Skipping.")
        continue

    print(f"Collecting data for class {classLetterMapping(j)}")

    isSkip = False

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture image from webcam.")
            break

        cv2.putText(
            frame,
            'Press Q to collect or S to skip',
            (100, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow('frame', frame)

        key = cv2.waitKey(25) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            isSkip = True
            break

    if not ret:
        break

    if isSkip:
        print(f"Skipping class {classLetterMapping(j)}")
        continue

    # Resume from the number of existing images
    counter = len(images)

    while counter < dataset_size:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture image from webcam.")
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        image_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(image_path, frame)

        counter += 1

    if not ret:
        break

    print(f"Class {classLetterMapping(j)} completed!")

cap.release()
cv2.destroyAllWindows()
