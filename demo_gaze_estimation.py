import cv2
import mediapipe as mp
import numpy as np
import time
import math

left_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=20  # Adjust to allow detection of multiple faces
)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

attention_duration = 5  # Duration to track head pose for attention calculation (in seconds)
attention_threshold = 0.5  # Threshold to determine attention (e.g., 50% looking in the same direction)

eye_closed_time = None  # Time when eyes were closed
sleeping_duration = 3  # Duration to track closed eyes for attention calculation (in seconds)
# attention_threshold = 0.5  # Threshold to determine attention (e.g., 50% looking in the same direction)

# Initialize data structures for tracking head pose directions
face_directions = []
start_time = time.time()

def distance(point1, point2):
    x1, y1 =  point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def blinkRatio (image, landmarks, right_eye_indices, left_eye_indices):
    rh_right = landmarks[right_eye_indices[0]]
    rh_left = landmarks[right_eye_indices[8]]

    rv_top = landmarks[right_eye_indices[12]]
    rv_bottom = landmarks[right_eye_indices[4]]

    lh_right = landmarks[left_eye_indices[0]]
    lh_left = landmarks[left_eye_indices[8]]

    lv_top = landmarks[left_eye_indices[12]]
    lv_bottom = landmarks[left_eye_indices[4]]

    cv2.line(image, (int(rh_right.x * image.shape[1]), int(rh_right.y * image.shape[0])), (int(rh_left.x * image.shape[1]), int(rh_left.y * image.shape[0])), (0, 255, 0), 2)
    cv2.line(image, (int(rv_top.x * image.shape[1]), int(rv_top.y * image.shape[0])), (int(rv_bottom.x * image.shape[1]), int(rv_bottom.y * image.shape[0])), (0, 255, 0), 2)
    cv2.line(image, (int(lh_right.x * image.shape[1]), int(lh_right.y * image.shape[0])), (int(lh_left.x * image.shape[1]), int(lh_left.y * image.shape[0])), (0, 255, 0), 2)
    cv2.line(image, (int(lv_top.x * image.shape[1]), int(lv_top.y * image.shape[0])), (int(lv_bottom.x * image.shape[1]), int(lv_bottom.y * image.shape[0])), (0, 255, 0), 2)

    rh_distance = distance((rh_right.x * image.shape[1], rh_right.y * image.shape[0]), (rh_left.x * image.shape[1], rh_left.y * image.shape[0]))
    rv_distance = distance((rv_top.x * image.shape[1], rv_top.y * image.shape[0]), (rv_bottom.x * image.shape[1], rv_bottom.y * image.shape[0]))

    lh_distance = distance((lh_right.x * image.shape[1], lh_right.y * image.shape[0]), (lh_left.x * image.shape[1], lh_left.y * image.shape[0]))
    lv_distance = distance((lv_top.x * image.shape[1], lv_top.y * image.shape[0]), (lv_bottom.x * image.shape[1], lv_bottom.y * image.shape[0]))

    rh_ratio = rh_distance / rv_distance
    lh_ratio = lh_distance / lv_distance

    ration = (rh_ratio + lh_ratio) / 2
    return ration



cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    start = time.time()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    text_position = 50
    direction_counts = {'Forward': 0, 'Looking left': 0, 'Looking right': 0, 'Looking up': 0, 'Looking down': 0}

    if results.multi_face_landmarks:

        current_time = time.time()
        elapsed_time = current_time - start_time

        for face_landmarks in results.multi_face_landmarks:

            img_h, img_w, img_c = image.shape
            face_3d = []
            face_2d = []
            
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx==291 or idx ==  199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append((x, y))
                    face_3d.append((x, y, lm.z))

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            ratio = blinkRatio(image, face_landmarks.landmark, right_eye_indices, left_eye_indices)
            # print(ratio) 
            if ratio > 3.8:
                # print("Blinking")
                if eye_closed_time is None:
                    eye_closed_time = time.time()
                elif time.time() - eye_closed_time >= sleeping_duration:
                    print("Sleeping")
            else:
                eye_closed_time = None  # Reset eye closure       

            focal_length = 1 * image.shape[1]
            cam_matrix = np.array([[focal_length, 0, image.shape[1]/2],
                                    [0, focal_length, image.shape[0]/2],
                                    [0, 0, 1]])

            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix)

            rmat, _ = cv2.Rodrigues(rot_vec)

            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            face_directions.append((x, y, z))
    
            if y < -10:
                text = "Looking left"
            elif y > 10:
                text = "Looking right"
            elif x < -10:
                text = "Looking down"
            elif x > 10:
                text = "Looking up"
            else:
                text = "Forward"
            
            direction_counts[text] += 1    

            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)
            num_faces = str(len(results.multi_face_landmarks)) 

            total_faces = len(results.multi_face_landmarks)
            percentage_dict = {key: (value / total_faces) * 100 for key, value in direction_counts.items()}


            # cv2.putText(image, text, (20, text_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(image, "Num of faces: " + num_faces, (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            text_position += 50

            text_position2 = 50
            text_spacing = 30

            # cv2.rectangle(image, (0, 0), (250, text_position2 + len(percentage_dict) * text_spacing), (0, 0, 0), -1) 
            # for direction, percentage in percentage_dict.items():
            #     text = f"{direction}: {percentage:.1f}%"
            #     cv2.putText(image, text, (20, text_position2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            #     text_position2 += text_spacing  

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    thickness=1, circle_radius=1)
            )

    end = time.time()
    totalTime = end - start

    cv2.imshow('Demo - gaze estimation', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()