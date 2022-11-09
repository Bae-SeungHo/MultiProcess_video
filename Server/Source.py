from socket import *
import cv2
import numpy as np

import mediapipe as mp
import os
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import math

global Server , client
Server = None
client = None

def Create(PORT):
    global Server
    Server = socket(AF_INET,SOCK_STREAM)
    Server.bind(('',PORT))
    Server.listen(2)

def Close():
    global Server , client
    client.close()
    
def Accept():
    global Server , client
    client, addr = Server.accept()
    return client

def recvall(sock, count):
	buf = b''
	while count:
		newbuf = sock.recv(count)
		if not newbuf :
			return None
		buf += newbuf
		count -= len(newbuf)
	return buf


def cal_Radian(handlist,p2,p1):
    p=[]
    rad_list=[]
    handlist = np.delete(handlist,2,axis=1)
    for i in range(len(p2)):
        temp = handlist[p2[i]] - handlist[p1[i]]
        p.append(temp)
    
    for i in p:
        rad = math.atan2(i[1],i[0])
        rad_list.append(rad)
    return rad_list


def R_Thread(client,Q,Online):
    print('R_Thread Start')
    sound = str("b'RIFF$")
    filename='Record/record.wav'
    data_transferred = 0
    while True:
        length = recvall(client, 16)
        if str(length).startswith(sound):
            print('yes')
            with open(filename, 'wb') as f: #현재dir에 filename으로 파일을 받는다
                try:
                    while length: #데이터가 있을 때까지
                        
                        f.write(length) #1024바이트 쓴다
                        data_transferred += len(length)
                        length = client.recv(1024) #1024바이트를 받아 온다
                        print(length)
                        if length[-2:]==b'sd':
                            f.write(length[:-2])
                            break
                except Exception as ex:
                    print(ex)
            print('파일 %s 받기 완료. 전송량 %d' %(filename, data_transferred))
            data_transferred = 0
        else:
            stringData = recvall(client, int(length))
            data = np.frombuffer(stringData, dtype='uint8')
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            cv2.imshow('RecivedFrame', frame)
            Q.put(frame)
            #self.FrameQ.put(frame)
            cv2.waitKey(1)

            if not Online.empty():
                print('Server_R_Thread_OFF')
                break
                 
def T_Thread(client,Q,Online):
    print('T_Thread Start')
    while True:
        if not Q.empty():
            val = int(Q.get())
            #argmax = val.index(max(val))
            client.send(val.to_bytes(2,byteorder='little'))
            
        if not Online.empty():
            argmax = 0
            client.send(argmax.to_bytes(2,byteorder='little'))
            print('Server_T_Thread_OFF')
            break
            

            
def M_Thread(client,FQ,GQ,OQ):
    print('M_Thread Start')
    #Init
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    #model = load_model('Model/L_model.h5')
    options = ['None','Call','Picture']
    pose_landmark = [0,11,12,13,14,15,16,23,24,25,26,27,28]

    #Threshold
    ACCURACY_THRESHOLD = 0.8
    LANDMARK_NOISE_THRSHOLD = 10
    NOISE = np.exp(-10)
    MAX_NOISE = np.exp(10)
    DIFF_NOISE = 0.5
    PREDICT_START_THRESHOLD = 50

    #Param
    seq_length = 30
    model = load_model('hand_model.h5')
    Turn = None
    
    Video_index = 0
    Pred = False
    Err_stack = 0
    
    R_hand_list = np.empty((1,24,3),dtype='float32')
    L_hand_list = np.empty((1,24,3),dtype='float32')
    R_hand_past = np.empty([21,3],dtype='float32')
    L_hand_past = np.empty([21,3],dtype='float32')
    R_list = np.empty((1,21,3),dtype='float32')
    L_list = np.empty((1,21,3),dtype='float32')
    pred_list = [0 for i in range(len(options))]
    with mp_holistic.Holistic(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.5) as holistic:
        while True:
            if FQ.qsize() > 15:
                FQ.get()
                FQ.get()
                FQ.get()
            elif FQ.qsize() > 5:
                FQ.get()
                FQ.get()
            elif FQ.qsize() > 3:
                FQ.get()
                
            
            if not FQ.empty():
                if Err_stack >= 5:
                    R_hand_list = np.empty((1,24,3),dtype='float32')
                    L_hand_list = np.empty((1,24,3),dtype='float32')
                    R_list = np.empty((1,21,3),dtype='float32')
                    L_list = np.empty((1,21,3),dtype='float32')
                    Err_stack = 0
                Video_index += 1
                frame = FQ.get()
                
                if Video_index == 1:
                    Frame_X = np.array(frame).shape[1]
                    Frame_Y = np.array(frame).shape[0]  
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                GO = False
                mp_drawing.draw_landmarks(frame,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(frame,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)

                R_hand = np.empty([1,3],dtype='float32')
                if results.right_hand_landmarks:
                    R_hand_center = (results.right_hand_landmarks.landmark[9].x ,results.right_hand_landmarks.landmark[9].y,results.right_hand_landmarks.landmark[9].z) # 

                    for i,point in enumerate(results.right_hand_landmarks.landmark):
                        a = np.array((point.x,point.y,point.z)).reshape([1,3])
                        R_hand = np.vstack((R_hand,a))

                        if point.x < NOISE or point.y < NOISE:
                            print('::ErrorCode1::',end=' ')
                            break 
                        if point.x > 1 or point.y > 1 or point.z > 1:
                            print('::ErrorCode2::',end=' ')
                            break
                        if not np.exp(point.x) or not np.exp(point.y) or not np.exp(point.z):
                            print('::ErrorCode3::',end=' ')
                            break

                    if len(R_hand) != 1:
                        R_hand = np.delete(R_hand,0,axis=0) 
                        if len(R_hand) != 21 or R_hand.max() > LANDMARK_NOISE_THRSHOLD:
                            print('R_Shape_Error , Shape Len : {}'.format(len(R_hand)))
                            Err_stack += 1
                            continue
                        if not Video_index:
                            a = R_hand*100
                        else:
                            a = (R_hand - R_hand_past)*100
                        R_hand_past = R_hand
                        if abs(np.mean(a)) > 5:
                            print('::ErrorCode4:: , mean : {}'.format(np.mean(a)))
                        else:
                            b = cal_Radian(R_hand,(6,10,14,18,7,11,15,19,1),(5,9,13,17,6,10,14,18,0))
                            b = np.array(b).reshape(3,3)
                            a = np.vstack((a,b))
                            a = a.reshape([1,24,3])
                            R_hand_list = np.vstack((R_hand_list,a))

                            R_hand = R_hand.reshape([1,21,3])
                            R_list = np.vstack((R_list,R_hand))
                            GO = True

                L_hand = np.empty([1,3],dtype='float32')
                if results.left_hand_landmarks:
                    L_hand_center = (results.left_hand_landmarks.landmark[9].x ,results.left_hand_landmarks.landmark[9].y,results.left_hand_landmarks.landmark[9].z)
                    for i,point in enumerate(results.left_hand_landmarks.landmark):
                        a = np.array((point.x,point.y,point.z)).reshape([1,3])
                        L_hand = np.vstack((L_hand,a))
                        if point.x < NOISE or point.y < NOISE:
                            print('::ErrorCode1::',end=' ')
                            break 
                        if point.x > 1 or point.y > 1 or point.z > 1:
                            print('::ErrorCode2::',end=' ')
                            break
                        if not np.exp(point.x) or not np.exp(point.y) or not np.exp(point.z):
                            print('::ErrorCode3::',end=' ')
                            break

                    if len(L_hand) != 1:
                        L_hand = np.delete(L_hand,0,axis=0)
                        if len(L_hand) != 21 or L_hand.max() > LANDMARK_NOISE_THRSHOLD:
                            print('L_Shape_Error , Shape Len : {}'.format(len(L_hand)))
                            Err_stack += 1
                            continue
                        if not Video_index:
                            a = L_hand*100
                        else:
                            a = (L_hand - L_hand_past)*100
                        L_hand_past = L_hand

                        if abs(np.mean(a)) > 5:
                            print('::ErrorCode4:: , mean : {}'.format(np.mean(a)))
                        else:
                            b = cal_Radian(L_hand,(6,10,14,18,7,11,15,19,1),(5,9,13,17,6,10,14,18,0))
                            b = np.array(b).reshape(3,3)
                            a = np.vstack((a,b))
                            a = a.reshape([1,24,3])
                            L_hand_list = np.vstack((L_hand_list,a))

                            L_hand = L_hand.reshape([1,21,3])
                            L_list = np.vstack((L_list,L_hand))
                            LGO = True
                #endif
                Err_stack = 0
                if len(R_hand_list) > len(L_hand_list) :
                    Dhand = R_hand_list
                    Turn = 'Right'
                else:
                    Dhand = L_hand_list
                    Turn = 'Left'
                #endif

                # Prediction
                if len(Dhand) > PREDICT_START_THRESHOLD:
                    predictable = False

                    if Turn == 'Right':
                        if len(Dhand) == len(R_list):
                            Dhand = np.hstack((Dhand,R_list))
                            predictable = True    
                    else:
                        if len(Dhand) == len(L_list):
                            Dhand = np.hstack((Dhand,L_list))
                            predictable = True
                    if predictable:

                        input_data = np.array(Dhand[-seq_length:]).reshape([-1,seq_length,135])
                        pred = model.predict(input_data)
                        one = pred.argmax()

                        if pred[0][one] > ACCURACY_THRESHOLD:
                            #cv2.putText(frame,'Predict Action : %s' % (options[one]),(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
                            print('input Gesture Q',options[one],pred,one)
                            if one:
                                GQ.put(one)

                    R_hand_list = np.empty((1,24,3),dtype='float32')
                    L_hand_list = np.empty((1,24,3),dtype='float32')
                    R_list = np.empty((1,21,3),dtype='float32')
                    L_list = np.empty((1,21,3),dtype='float32')

                cv2.putText(frame,str(R_hand_list.shape[0]),(20,Frame_Y-60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                cv2.putText(frame,str(L_hand_list.shape[0]),(20,Frame_Y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                cv2.putText(frame,str(R_list.shape[0]),(60,Frame_Y-60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                cv2.putText(frame,str(L_list.shape[0]),(60,Frame_Y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

                cv2.imshow('frame', frame)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    break