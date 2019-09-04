import face_alignment
import cv2
import time
poczatek=time.time()
import numpy as np
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,flip_input=False,device='cuda')
input = cv2.imread("kupa.png")
preds = fa.get_landmarks(input)

# lista=[]
# for i in range(len(preds[0])):
#     lista.append((preds[0][i][0],preds[0][i][1]))
# print(type(lista[0]))
koniec=time.time()
print(koniec-poczatek)
for i in range(len(preds[0])):
    cv2.circle(input,(preds[0][i][0],preds[0][i][1]),1,(0,0,0))
cv2.imshow('kozak',input)
cv2.waitKey(0)
