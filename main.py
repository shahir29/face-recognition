import cv2
import face_recognition
import numpy as np

# Load the reference images
reference_image1 = face_recognition.load_image_file("shahir.jpg")
reference_encoding1 = face_recognition.face_encodings(reference_image1)

reference_image2 = face_recognition.load_image_file("reference2.jpg")
reference_encoding2 = face_recognition.face_encodings(reference_image2)

# Ensure encodings were found
if len(reference_encoding1) == 0 or len(reference_encoding2) == 0:
    print("Error: Could not encode faces from reference images.")
    exit()

# Store encodings and corresponding labels
known_encodings = [reference_encoding1[0], reference_encoding2[0]]
known_labels = ["Shahir", "Shakeel"]

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Accuracy tuning parameters
TOLERANCE = 0.4  # Lower means stricter matching
FRAME_RESIZE_FACTOR = 0.5  # Resize frame to speed up processing
MATCH_THRESHOLD = 3  # Require multiple consecutive matches to confirm recognition
match_counter = {}  # Track match confidence

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_FACTOR, fy=FRAME_RESIZE_FACTOR)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")  # More accurate than 'hog'
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Scale face locations back to original frame size
        top, right, bottom, left = [int(coord / FRAME_RESIZE_FACTOR) for coord in [top, right, bottom, left]]

        # Compare with known encodings
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)  # Get the closest match

        # Check if the best match is within the tolerance
        if face_distances[best_match_index] < TOLERANCE:
            name = known_labels[best_match_index]
            color = (0, 255, 0)

            # Update confidence counter
            if name in match_counter:
                match_counter[name] += 1
            else:
                match_counter[name] = 1

            # Only confirm match if seen multiple times
            if match_counter[name] >= MATCH_THRESHOLD:
                label = f"MATCH: {name}"
            else:
                label = "Scanning..."

        else:
            name = "NO MATCH"
            color = (0, 0, 255)
            label = "NO MATCH!"

        # Display the result
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Show the frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()