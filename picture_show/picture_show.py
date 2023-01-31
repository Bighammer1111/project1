import os
from random import shuffle
import time
import Play_mp3
import cv2
import numpy as np
from itertools import cycle
from pygame import mixer
from text import cv2ImgAddText

path_animal='D://emotion_database//A'
path_human='D://emotion_database//H'
path_normal='D://emotion_database//N'
path_positive='D://emotion_database//P'
path_snack="D://emotion_database//Sn"
path_spider='D://emotion_database//Sp'

frame_path=[path_positive,path_animal,path_normal]#設定播放路徑
music_path=['positive.mp3','negetive.mp3','natual.mp3']#播放音樂檔
picture_second=6000  #每張圖60秒
picture_count=30  #30張圖片

for i in range(0,3):
    file_name=os.listdir(frame_path[i])#讀取資料夾内所有檔名
    shuffle(file_name)#打亂順序
    count=0

    img_iter = cycle([cv2.imread(os.sep.join([frame_path[i] , x]))for x in file_name])#設定依序播放

    key = 0
    count=0
    mixer.init()
    mixer.music.load(music_path[i])
    mixer.music.play()
    # while key & 0xFF != 27:
    #     if count>=picture_count:
    #         break
    #     cv2.namedWindow('yyy', cv2.WINDOW_NORMAL)
    #     cv2.setWindowProperty('yyy', cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    #     cv2.imshow('yyy',next(img_iter))
    #     key = cv2.waitKey(picture_second)#每隔6秒換一次圖片
    #     count=count+1
    image =np.zeros((1920, 1080, 3), np.uint8)
    text='音樂播放'
    cv2.namedWindow('yyy', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('yyy', cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    image = cv2ImgAddText(image, text, 400, 600, (255, 255, 255), 100)
    cv2.imshow('yyy',image)
    key = cv2.waitKey(3*60000)
    mixer.music.fadeout(5000)

    image =np.zeros((1920, 1080, 3), np.uint8)
    text='休息時間'
    image = cv2ImgAddText(image, text, 400, 600, (255, 255, 255), 100)
    cv2.imshow('yyy',image)
    key = cv2.waitKey(6*60000)

