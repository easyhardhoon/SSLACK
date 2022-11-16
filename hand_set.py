import cv2
import mediapipe as mp
import numpy as np

max_num_hands = 1 # 인식할 수 있는 손 개수
gesture = {
        0:'k_1', 1:'k_2', 2:'k_3', 3:'k_4', 4:'k_5', 5:'k_6',
        6:'k_7', 7:'k_8', 8:'k_9', 9:'k_10', 10:'k_11', 11:'k_12', 12:'k_13', 
        13:'k_8'
        } # 12가지의 제스처, 제스처 데이터는 손가락 관절의 각도와 각각의 라벨을 뜻한다.

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

# Gesture recognition data
#file = np.genfromtxt('data/train.csv', delimiter=',')
file = np.genfromtxt('data/train_new.csv', delimiter=',')
#file = None
#print(file.shape)

cap = cv2.VideoCapture(0)

def click(event, x, y, flags, param):
    global data, file
    if event == cv2.EVENT_LBUTTONDOWN:
        #print(data.shape) ---> 16. (15 numbers of angle point + @label)
        file = np.vstack((file, data)) # error......to do....if ... data/gesture_train.csv down
        #print(file)
        print(file.shape)

cv2.namedWindow('Dataset')
cv2.setMouseCallback('Dataset', click)
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints ...
            # parent joint ---> [] - edge joints (4,8,12,16,20)
            # child  joint ---> [] - "0" (base joint) (0)
            # so... v2 -v1 's result is exact vector array
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3] ....... for each joint by joint 's internal vector

            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            # angle : 15... 3+3+3+3+3 ...!!!! 
            # probably... that's not enough to represent 32 labels....
            # about ... apply hoon's last algorithm... use x,y pixels.
            # just +@ .... use this method main and sub-use hard-coding x&y pisxel
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree
            data = np.array([angle], dtype=np.float32)
            # ---------------------------------------------------------------
            #NOTE 221031
            if(res.landmark[mp_hands.HandLandmark.WRIST].x < res.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x):
                if(res.landmark[mp_hands.HandLandmark.WRIST].y > res.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y):
                    falm_state = 1 # to 12'
            if(res.landmark[mp_hands.HandLandmark.WRIST].x > res.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x):
                if(res.landmark[mp_hands.HandLandmark.WRIST].y > res.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y):
                    falm_state = 2 # to 9'
            if(res.landmark[mp_hands.HandLandmark.WRIST].x > res.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x):
                if(res.landmark[mp_hands.HandLandmark.WRIST].y < res.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y):
                    falm_state = 3 # to 6'
            if(res.landmark[mp_hands.HandLandmark.WRIST].x < res.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x):
                if(res.landmark[mp_hands.HandLandmark.WRIST].y < res.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y):
                    falm_state = 4 # t 3' --> actually NONE
            data = np.append(data,falm_state*10) # falm_state's weight "10"
            # -----------------------------------------------------------------
            data = np.append(data,5) # label NOTE 
            for i in range(data.size):
                data[i] = round(data[i],5)
            # ---------------------------------------------------------------
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
            #np.savetxt('data/train.csv', file, delimiter=',')
            np.savetxt('data/train_new.csv', file, delimiter=',')
    cv2.imshow('Dataset', img)
    if cv2.waitKey(1) == ord('q'):
        break

#np.savetxt('data/gesture_train_fy.csv', file, delimiter=',')
