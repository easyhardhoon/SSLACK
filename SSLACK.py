import cv2
import mediapipe as mp
import numpy as np

# mediapipe_algo setup code

max_num_hands = 2
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

file = np.genfromtxt('data/train.csv', delimiter=',') # ---> demo csv [train.csv]
#file = np.genfromtxt('data/train_new.csv', delimiter=',')  # ---> updated csv [train_new.csv]
# ---------------------
# NOTE
# what's new ---> updated "palm state" which is used as "extra" parameter
# [15], [1] --> [15,1],[1]
# palm_1st, palm_2nd, palm_3rd, palm_4th 
# use hand_landmarks.landmark[mp.hands.HandLandmark.WRIST].x, .y
#     hand_landmarks.landmark[mp.hands.HandLandmark.INDEX_FINGER_MCP].x , .y
# 
# ---------------------

angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label) 

# CNN_algo setup code
# TODO


# ----------------------------------------------------------------------------
# NOTE "below 3 functions" are our main point

def mediapipe_algo(res,img):
    mp_result = None
    joint = np.zeros((21, 3)) #1~20 landmark * (x,y,z) pixel
    for j, lm in enumerate(res.landmark):
        joint[j] = [lm.x, lm.y, lm.z]

    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
    v = v2 - v1 # [20,3]
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    angle = np.arccos(np.einsum('nt,nt->n',
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

    angle = np.degrees(angle) # Convert radian to degree
    data = np.array([angle], dtype=np.float32)
    ret, results, neighbours, dist = knn.findNearest(data, 5)
    idx = int(results[0][0])

    korean_dict = {0:"ㄱ", 1:"ㄴ",2:"권승찬바보", 3:"ㄷ" , 4: "ㄹ", 5:"ㅁ",6:"ㅂ", 7:"ㅅ", 8:"ㅇ", 9:"ㅈ",10:"ㅊ",
            11:"ㅌ",12:"ㅍ", 13: "ㅎ", 14 : "end",
            15: "ㅏ", 16: "ㅑ", 17:"ㅓ", 18: "ㅕ", 19:"ㅗ", 20:"ㅛ", 21: "ㅜ", 22: " ㅠ",
            23: "ㅡ", 24: "ㅣ"}
    if(0<= idx <=24) : 
        print(korean_dict[idx])
        mp_result = korean_dict[idx]
    else:
        print("error")
        mp_result = "error"
    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
    return mp_result

def cnn_algo(img):
    cnn_result = None
    # -------------------------------------
    #TODO : make cnn_algo
    #     -->  run cnn_based model
    #     -->  return cnn_result
    # -------------------------------------
    return cnn_result

def ensemble(mp_result, cnn_result):
    final_result = None
    # -------------------------------------
    #TODO : make ensemble code
    #     --> how to rate two's
    #     --> use "accuracy" or "queue" .....
    #NOTE : maybe use queue
    # -------------------------------------
    return final_result
# --------------------------------------------------------------------------------------


final_words_queue = []
for i in range(10):
    final_words_queue.append("empty")

cap = cv2.VideoCapture(0)

# main 
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # based on mediapipe's logic flow
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            mp_result = mediapipe_algo(res,img)
            cnn_result = cnn_algo(img)
            final_result = ensemble(mp_result, cnn_result)
            final_result = "test"
            # --------------------------------------------------------------------
            # FIXME--> maybe make real_queue(or list) by using final_words_queue based on new rule
            #      --> ex) continuous "3" same data based on "queue"
            # NOTE --> should decide between "real_queue" or "timer"  
            final_words_queue.pop(0)  # pop first data
            final_words_queue.append(mp_result) # append "final_result" data

            # NOTE ===> if detected hand's num == 2 : --> save "now" label to real_queue and pass
            if(max_num_hands ==2): # not max_num_hands ....
                pass
                #final_words_queue.clear()
                #time.sleep(1) 
                # NOTE ===> pass next step smoothly
                #      ===> minimizae queue size
            # --------------------------------------------------------------------
            if(mp_result == "END"):
                # ----------------------------------------------------------------
                # when "end motion" detected ....
                # TODO ==> 1. convert final_word_queue to final_sentence 
                #          2. make exact final_sentence by using etc) method
                #          3. return SSLACK_message 
                #             -----> to allow for AI_speaker to detect SSLACK_message 
                #          4. run AI_speaker
                # ---------------------------------------------------------------
                # FIXME ==> update please .......
                # ----------------------------------------------------------------
                print("time to end")
                break
    cv2.imshow('SSLACK', img)
    if cv2.waitKey(1) == ord('q'):
        break
