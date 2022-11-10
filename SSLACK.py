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

#file = np.genfromtxt('data/train.csv', delimiter=',') # ---> demo csv [train.csv]
file = np.genfromtxt('data/train_new.csv', delimiter=',')  # ---> updated csv [train_new.csv]
# [15],[1] ==> [15,1],[1]

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
    joint = np.zeros((21, 3)) 
    for j, lm in enumerate(res.landmark):
        joint[j] = [lm.x, lm.y, lm.z]

    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] 
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] 
    v = v2 - v1 
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    angle = np.arccos(np.einsum('nt,nt->n',
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) 

    angle = np.degrees(angle) 
    data = np.array([angle], dtype=np.float32)
    # ------------------------------------------------
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

    angle = np.append(angle,falm_state*10) # falm_sate's weight 10
    #data = np.append(data, falm_state*10) # @@@@ 
    data = np.array([angle], dtype = np.float32)
    #---------------------------------------------------------
    ret, results, neighbours, dist = knn.findNearest(data, 5)
    idx = int(results[0][0])

    korean_dict = {0:"ㄱ", 1:"ㄴ",2:"ㄷ", 3:"ㄹ" , 4: "ㅁ", 5:"END",6:"ㅂ", 7:"ㅅ", 8:"ㅇ", 9:"ㅈ",10:"ㅊ",
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


words_queue = []
for i in range(10):
    words_queue.append("NULL")

final_list = []
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
        detected_hand_N = len(result.multi_hand_landmarks)
        for res in result.multi_hand_landmarks:
            mp_result = mediapipe_algo(res,img)
            cnn_result = cnn_algo(img)
            final_result = ensemble(mp_result, cnn_result)
            final_result = mp_result
            words_queue.pop(0)  
            words_queue.append(final_result)
            # NOTE ===> if detected_hand_N == 2 : --> save "now" label to final_list and pass
            if(detected_hand_N >1):
                 # --------------------------------------------------------------------
                 # FIXME--> maybe make final_list by using words_queue based on new rule
                 #      --> ex) continuous "3" same data based on "queue"
                 #      --> NOTE solved by "find max counted value"
                #for word in words_queue:
                    #final_list.append(max(set(words_queue, key=words_queue.count))) 
                M = max(words_queue, key = words_queue.count)
                if M != "NULL":
                    final_list.append(M)
                print("final list : ", final_list)
                words_queue.clear()
                for i in range(10):
                    words_queue.append("NULL")
                #time.sleep(1) #FIXME timer....???
                # ===> pass next step smoothly
                # ===> minimize queue size
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
                # FIXME ==> update please .......use final_list
                # ----------------------------------------------------------------
                final_length = len(final_list)
                print("time to end .... run AI-SPEAKER")
                break
    cv2.imshow('SSLACK', img)
    if cv2.waitKey(1) == ord('q'):
        break
