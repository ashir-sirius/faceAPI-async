import face_recognition
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import numpy
import tensorflow as tf

import asyncio

#img1_path = 'C:/Users/HP/Desktop/Sirius-Support/models/image.jpg'
#img2_path = 'C:/Users/HP/Desktop/Sirius-Support/models/image-d.jpg'


async def detect_face(img1_path, img2_path):
    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=160, margin=0)

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    confidence = []

    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    j = 1
    img_cropped = mtcnn(
        img1, save_path=f'C:/Users/HP/Desktop/models/frame_{j}_face.jpg')
    embd1 = resnet(img_cropped.unsqueeze(0))

    j = 2
    img_cropped = mtcnn(
        img2, save_path=f'C:/Users/HP/Desktop/models/frame_{j}_face.jpg')
    embd2 = resnet(img_cropped.unsqueeze(0))

    embd1 = embd1.detach().numpy()
    embd2 = embd2.detach().numpy()

    # print(type(embd1))
    # print(type(embd2))

    matches = face_recognition.compare_faces(embd1, embd2, 0.8)
    # print(matches)

    if True in matches:
        return "Faces are of same person"
    else:
        return "Faces are of different person"


#boola = detect_face(img1_path, img2_path)
#if boola:
#    print('faces are of same person')
#else:
#    print('faces are of different person')
