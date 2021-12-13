# -*- coding: utf-8 -*-
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

import socket
import os

import base64
import numpy as np
from inference_full import Final_inference
import cv2


# cnu_config.py
BINARY_CLASSIFICATION = '0x20'            # 0x20(32)
MULTI_CLASSIFICATION = '0x21'             # 0x21(33)
OBJECT_DETECTION = '0x22'                 # 0x22(34)
GAN_PIXELART = '0x23'                     # 0x23(35)
REGRESSION_PRICE = '0x24'                 # 0x24(36)


STATE_SEND = '0x00'

VALUE_NONE = '0x000x00'
VALUE_NO_1 = '0x000x01'
VALUE_NO_2 = '0x000x02'
VALUE_NO_3 = '0x000x03'


def processFunction(receive_data):
    print('----Received data: ' + str(receive_data))
    function_code = receive_data[0:4]  # 0x00 function-code
    state = receive_data[4:8]          #
    value = receive_data[8:]           #
    print('Function code: ' + str(function_code))

#socket 수신 버퍼를 읽어서 반환하는 함수
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def receive_img(sock, count):
    length = recvall(sock, count)
    length1 = length.decode('utf-8')
    stringData = recvall(sock, int(length1))
    data = np.frombuffer(base64.b64decode(stringData), np.uint8)
    decimg = cv2.imdecode(data, 1)
    print('이미지 수신 완료')
    return decimg

def send_img(sock, img):
    img_encode = cv2.imencode('.jpg', img)[1]
    data_encode = np.array(img_encode)
    stringData = base64.b64encode(data_encode)
    length = str(len(stringData))
    sock.sendall(length.encode('utf-8').ljust(64))
    sock.send(stringData)
    print('이미지 전송 완료')


def main():
    HOST = '168.131.151.38'       # 접속할 서버 주소
    PORT = 200                     # 클라이언트 접속을 대기하는 포트 번호
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)        # 소켓 객체 생성(IPv4, TCP 타입)
    serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)      # WinError 10048 에러 해결을 위한 코드
    print('Socket created')
    serverSocket.bind((HOST, PORT))             # 생성한 소켓에 설정한 HOST와 PORT 맵핑
    print('Socket bind complete')

    try:
        while True:
            serverSocket.listen(5)  # 맵핑된 소켓을 연결 요청 대기 상태로 전환
            # check network sync
            try:
                clientSocket, addr = serverSocket.accept()
                print("connection from: ", addr)

                # Receive request from client
                while True:
                    receive_data = clientSocket.recv(1024)              # 데이터 수신, 최대 1024
                    receive_data = receive_data.decode('utf-8')
                    function_code = receive_data[0:4]  # 0x00 function-code
                    state = receive_data[4:8]          #
                    value = receive_data[8:]           #
                    print('received function code: ',function_code)

                    
                     # MULTI_CLASSIFICATION
                    if (function_code == MULTI_CLASSIFICATION):        # 0x21(33)
                        sendBuf = function_code + STATE_SEND + VALUE_NONE
                        clientSocket.send(sendBuf.encode())

                        decimg = receive_img(clientSocket, 64)
                        cv2.imwrite('from_client.jpg', decimg)

                        # AI_MODEL
                        test_transform = transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

                        img = Image.open("from_client.jpg")
                        classifier = Final_inference(mode="classification", ckpt_path="weights/QC.pt", num_classes=3, transform=test_transform)

                        quality = classifier.classify_QC(img)
                        print(quality)
                       
                        if quality == "Low-quality":
                            value = VALUE_NO_1
                        elif quality == "Normal-quality":
                            value = VALUE_NO_2
                        elif quality == "High-quality":
                            value = VALUE_NO_3
                       

                        # # 결과 이미지 전송
                        # send_img(clientSocket, result_img)

                        # 인공지능 판별 결과 전송
                        sendBuf = function_code + STATE_SEND + value
                        clientSocket.send(sendBuf.encode())
                        clientSocket.close()

                    # BINARY_CLASSIFICATION
                    elif (function_code == BINARY_CLASSIFICATION):        # 0x20(32)
                        sendBuf = function_code + STATE_SEND + VALUE_NONE
                        clientSocket.send(sendBuf.encode())
                        decimg = receive_img(clientSocket, 64)
                        cv2.imwrite('from_client.jpg', decimg)

                        # AI_MODEL
                        test_transform = transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

                        img = Image.open("from_client.jpg")
                        classifier = Final_inference(mode="binary_classification", ckpt_path="./weights/Binary.pt", num_classes=2, transform=test_transform)

                        quality = classifier.classify_binary(img)
                        print(quality)
                       
                        if quality == "Negative":
                            value = VALUE_NO_1
                        elif quality == "Positive":
                            value = VALUE_NO_2

                        # # 결과 이미지 전송
                        # send_img(clientSocket, result_img)

                        # 인공지능 판별 결과 전송
                        sendBuf = function_code + STATE_SEND + value
                        clientSocket.send(sendBuf.encode())
                        clientSocket.close()

                    # OBJECT_DETECTION
                    elif (function_code == OBJECT_DETECTION):        # 0x22(34)
                        sendBuf = function_code + STATE_SEND + VALUE_NONE
                        clientSocket.send(sendBuf.encode())

                        decimg = receive_img(clientSocket, 64)
                        cv2.imwrite('from_client.jpg', decimg)
  
                        # os.mkdir("./detect_results")
                        img = Image.open("from_client.jpg")
                        detector = Final_inference(mode="detection", ckpt_path="weights/Detection.pt")
                        
                        detected_img = detector.detection(img)
                       
                        # os.rmdir("./detect_results")
                        # 결과 이미지 전송
                        send_img(clientSocket, detected_img)

                        # 인공지능 판별 결과 전송
                        clientSocket.close()

                    # PIXELART
                    elif (function_code == GAN_PIXELART):        # 0x23(35)
                        sendBuf = function_code + STATE_SEND + VALUE_NONE
                        clientSocket.send(sendBuf.encode())

                        decimg = receive_img(clientSocket, 64)
                        cv2.imwrite('PIXELART_from_client.jpg', decimg)

                        # PIXELART1
                        result_img = pixelart.photo2pixelartbyCV2(decimg, reduce_size=(32, 32))

                        # PIXELART2
                        # AP = applepixelart.applePreprocessing(decimg)
                        # segment_img, result_img = AP.apple_detect(decimg)
                        # opencv_image = applepixelart.apple_pixelart(segment_img, reduce_size=(32, 32))

                        # 결과 이미지 전송
                        send_img(clientSocket, result_img)
                        clientSocket.close()

                    # REGRESSION_PRICE
                    elif (function_code == REGRESSION_PRICE):  # 0x23(35)
                        sendBuf = function_code + STATE_SEND + value
                        clientSocket.send(sendBuf.encode())

                        # regressor = Final_inference(mode="regression", ckpt_path="weights/Regression.pt")
                        regressor = Final_inference(mode="regression", ckpt_path="./weights/Regression.pt")
                        input = np.array([[np.random.randint(-10, 30)]])
                        # input = np.array([[int(value)]])
                        price = regressor.regression(input)
                        
                        
                        # 가격 예측
                        print('{} 지역의 예측 가격 {}원 입니다.'.format(value, str(int(price))))
                        sendBuf = function_code + STATE_SEND + str(int(price))
                        clientSocket.send(sendBuf.encode())
                        clientSocket.close()

            except:
                print("Exception occurrs")
                # serverSocket.close()
                print("Socket closed")
            finally:
                print(addr)
    except Exception as e:
        print(e)
    else:
        serverSocket.close()

if __name__ == '__main__':
    main()