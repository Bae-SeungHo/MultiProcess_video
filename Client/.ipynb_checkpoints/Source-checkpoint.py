from socket import *
import cv2
import numpy as np
import speech_recognition as sr
import pyglet
from PIL import ImageFont, ImageDraw, Image


def Connect(PORT):
    server= socket(AF_INET,SOCK_STREAM)
    server.connect(('165.229.187.226',PORT))
    return server

def T_Thread(server,Online,RQ):
    cam = cv2.VideoCapture(0)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 99]
    data_transferred = 0
    while True:
        if not RQ.empty():
            filename = RQ.get()
            print("파일 %s 전송 시작" %filename)
            with open(filename, 'rb') as f:
                try:
                    data = f.read(16) #1024바이트 읽는다
                    while data: #데이터가 없을 때까지
                        data_transferred += server.send(data) #1024바이트 보내고 크기 저장
                        data = f.read(1024) #1024바이트 읽음

                except Exception as ex:
                    print(ex)
            server.send('sd'.encode('utf-8')) # end literal
            print("전송완료 %s, 전송량 %d" %(filename, data_transferred))
            data_transferred = 0
            cv2.waitKey(3000)
        else:
            status,frame=cam.read()
            _, frame = cv2.imencode('.jpg', frame, encode_param)
            frame = np.array(frame)
            stringData = frame.tobytes()
            server.sendall((str(len(stringData))).encode().ljust(16) + stringData)


            if not Online.empty():
                online = Online.get()
                if online == 1:
                    print('Paused')
                    while True:
                        if Online.get() == 1:
                            print('Resume')
                            break
                elif online == 0:      
                    print('break')
                    break
            
def R_Thread(server,Q,Online):
    while True:
        data = server.recv(8)
        Data = int.from_bytes(data,byteorder='little')
        if not Data:
            Online.put(0)
            break
        elif Data == 1:
            if Q.empty():
                Q.put(Data) #Start
                
                Q.put(Data) #End
            


def Sound_Thread(Q,Online,RQ):
    
    font=ImageFont.truetype("fonts/gulim.ttc",40)
    img = np.full(shape=(512,512,3),fill_value=255,dtype=np.uint8)
    img_ = Image.fromarray(img)
    while True:
        _ = Q.get()
        '''
        Recorded = False
        img_ = Image.fromarray(img)
        draw = ImageDraw.Draw(img_)
        draw.text((130,200),'말씀해주세요',font=font,fill=(0,0,0))
        show = np.array(img_)
        #sound = pyglet.resource.media('src/tell.wav')
        #sound.play()
        print('start')
        cv2.imshow('notify',show)
        cv2.waitKey(1000)
        
        while not Recorded:
            try:
                Recorded = True
                r = sr.Recognizer()
                audio = get_speech(r)
                text = r.recognize_google(audio,language='ko-KR')
                print('1',end=' ')
            except:
                print('2')
                Recorded = False
                img_ = Image.fromarray(img)
                draw = ImageDraw.Draw(img_)
                draw.text((140,200),'다시 말씀해주세요',font=font,fill=(0,0,0))
                show = np.array(img_)   
                cv2.imshow('notify',show)
                #sound = pyglet.resource.media('src/tell_again.wav')
                #sound.play()
                cv2.waitKey(1000)
        img_ = Image.fromarray(img)
        draw = ImageDraw.Draw(img_)
        draw.text((20,200),text,font=font,fill=(0,0,0))
        show = np.array(img_)
        cv2.imshow('notify',show)
        
        with open('Record/record.wav', 'wb') as file:
            wav_data = audio.get_wav_data()
            file.write(wav_data)
        #sound = pyglet.resource.media('Record/record.wav')
        #sound.play()
        cv2.waitKey(6000)
        cv2.destroyAllWindows()
        cv2.waitKey(3000)
        '''
        RQ.put('Record/record.wav')
        
        _ = Q.get()
        
def get_speech(recognizer):
    # 마이크에서 음성을 추출하는 객체
    recognizer = sr.Recognizer()

    # 마이크 설정
    microphone = sr.Microphone(sample_rate=16000)

    # 마이크 소음 수치 반영
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("소음 수치 반영하여 음성을 청취합니다. {}".format(recognizer.energy_threshold))

    # 음성 수집
    with microphone as source:
        print("Say something!")
        result = recognizer.listen(source)
    
    return result

# 함수 호출부
