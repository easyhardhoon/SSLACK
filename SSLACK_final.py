import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image
import random
import phoneme_korean_v1 as p_k
import asyncio
import tensorflow as tf
from tensorflow import keras
import time
from py_hanspell.hanspell import spell_checker
# mediapipe_algo setup code

max_num_hands = 2
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7)

model = load_model("C:/Users/user/aiproject/final_Lee/SSLACK/cnn_model/final_modelv3.h5") #TEST
file = np.genfromtxt("C:/Users/user/aiproject/final_Lee/SSLACK/SSLACK.csv", delimiter=',') 
# [15],[1] ==> [15,1],[1]

angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label) 



borderless_label = ["ㅠ", ".", "ㅅ"]
count = 0

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
            falm_state = 4 # t 3' 

    angle = np.append(angle,falm_state*30) # falm_sate's weight 10
    data = np.array([angle], dtype = np.float32)
    #---------------------------------------------------------
    ret, results, neighbours, dist = knn.findNearest(data, 50)
    idx = int(results[0][0])

    korean_dict = {0:"ㄱ", 1:"ㄴ",2:"ㄷ", 3:"ㄹ" , 4: "ㅁ", 5:"ㅂ",6:"ㅅ", 7:"ㅇ", 8:"ㅈ", 9:"ㅊ",10:"ㅋ",11:"ㅌ",12:"ㅍ", 13: "ㅎ", 14 : ".",15: "ㅏ", 16: "ㅑ", 17:"ㅓ", 18: "ㅕ", 19:"ㅗ", 20:"ㅛ", 21: "ㅜ", 22: "ㅠ",23: "ㅡ", 24: "ㅣ", 25:"ㅐ", 26:"ㅔ", 27:"ㅚ",28:"ㅟ",29:"ㅒ", 30:"ㅖ",31:"ㅢ",32:"DEL"}
    if(0<= idx <=33) : 
        #print("mp : ", (korean_dict[idx]))
        mp_result = korean_dict[idx]
    else:
        print("error")
        mp_result = "error"
    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
    return mp_result


def cnn_algo(img, model):
    cnn_result = None
    global count
    #model.summary()
    # status , frame = cap.read()
    #print("........", status, frame)
    # img2 = img.copy()
    print(count)
    count += 1
    cv2.imwrite("C:/Users/user/aiproject/test2/test.png", img)  # 저장할 경로와 이름
    img = keras.preprocessing.image.load_img("C:/Users/user/aiproject/test2/test.png", target_size=(150, 150))
    #img = tf.keras.preprocessing.image.smart_resize(img, (150, 150), interpolation='bilinear')
    #img = cv2.resize(img,dsize=(150,150))

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    input_data = img_to_array(img)
    
    input_data = tf.expand_dims(input_data, 0)
    #input_data = np.expand_dims(input_data, axis=0)
    #input_data = preprocess_input(input_data)
    prediction = model.predict(input_data)
    # score = tf.nn.softmax(prediction[0])
    predicted_class = np.argmax(prediction[0])
    #print("predicted_class: ", predicted_class)
    print("prediction[0]: ",prediction[0])
    #print("prediction: ", prediction)
    
    # print(score)
    #print("...........", predicted_class)
    print("This image most likely belongs to {}".format(borderless_label[predicted_class]))
    
    cnn_result = borderless_label[predicted_class]
    return cnn_result

def ensemble(mp_result, cnn_result):
    #NOTE : just vote. between mediapipe & CNN ---> maybe 8:2 versus
    final_result = None
    mp_weight = 20
    cnn_weight = 80
    final_result = random.choices([mp_result, cnn_result], [mp_weight, cnn_weight])
    final_result = ''.join(map(str, final_result))
    return final_result

# --------------------------------------------------------------------------------------

words_queue = []
for i in range(20):
    words_queue.append("NULL")

final_list = []
cap = cv2.VideoCapture(0)

handN_queue = []
for i in range(5):
    handN_queue.append(0)

BOUNDARY = 10


#main
while cap.isOpened():
    ret, img = cap.read()
    img_cnn = img.copy()
    if not ret:
        continue
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    final_L = len(final_list)
    # based on mediapipe's logic flow
    if result.multi_hand_landmarks is not None:
        detected_hand_N = len(result.multi_hand_landmarks)
        handN_queue.pop(0)
        handN_queue.append(detected_hand_N)
        for res in result.multi_hand_landmarks:
            mp_result = mediapipe_algo(res,img)
            final_result = mp_result
            # ---------------------------------------------------------
            # NOTE only run CNN & ensemble when mp_result is in borderless_label
            if(mp_result in borderless_label):
                cnn_result = cnn_algo(img_cnn, model)
                final_result = ensemble(mp_result, cnn_result)
            print("final result : ", final_result)
            # ---------------------------------------------------------
            words_queue.pop(0) 
            words_queue.append(final_result)
            handN_sum = 0
            for N in handN_queue:
                handN_sum += N
            if(handN_sum >= BOUNDARY):
                M = max(words_queue, key = words_queue.count)
                if M == "DEL" and len(words_queue) > 0:
                    final_list.pop()
                    start_time = time.time()
                    while time.time() < start_time + 1:
                        pass
                if M != "NULL" and M != "DEL":
                    final_list.append(M)
                    start_time = time.time()
                    while time.time() < start_time + 1:
                        pass
                    
                words_queue.clear()
                for i in range(20):
                    words_queue.append("NULL")
                handN_queue.clear()
                for i in range(5):
                    handN_queue.append(0)
                handN_sum = 0
            specific_detector = max(words_queue, key = words_queue.count)
            if(specific_detector == "END"):
                # ---------------------------------------------------------------
                # FIXME ==> update sentence & AI_SPEAKER code. use final_list
                # ----------------------------------------------------------------
                print("time to end .... run AI-SPEAKER")
                #print("final list is : ", final_list)
                #time.sleep(10)
                break_point = False
                break
            if(len(final_list) != final_L):
                print("final_list : ",final_list)
    cv2.rectangle(img, (30, 350), (620,470), (255,255,255), -1)
    x_coordinate = 40
    y_coordinate = 350
    if len(final_list) >= 1:
        for i in final_list:
            font = ImageFont.truetype("fonts/gulim.ttc", 20)
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            if x_coordinate >= 600:
                x_coordinate = 40
                y_coordinate += 50
            draw.text((x_coordinate, y_coordinate), i ,font=font, fill=(0, 0, 0))
            img = np.array(img)
            x_coordinate += 40
    cv2.imshow('SSLACK', img)
    # --------------------------------------------
    # FIXME ==> print GUI sentence in Img
    # -------------------------------------------
    if cv2.waitKey(1) & 0xFF == ord('q'):
        ph = p_k.Phoneme_korean(final_list)
        ph.test()
        time.sleep(3)
        cv2.destroyWindow('SSLACK')
        break

#ph.merge_phoneme()
sentence_ph = ph.get_str()
font = ImageFont.truetype("BMJUA_ttf.ttf", 20)
img = np.full((200,300,3), (255,255,255), np.uint8)
img = Image.fromarray(img)
draw = ImageDraw.Draw(img)
draw.text((100,70), sentence_ph ,font=font, fill=(0, 0, 0))
img = np.array(img)
cv2.namedWindow("background", cv2.WINDOW_NORMAL)
cv2.resizeWindow(winname = "background", width=600, height=450)
cv2.imshow("background", img)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
