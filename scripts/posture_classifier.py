import os

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import _pickle as cPickle

from pose_encoder import FullBodyPoseEmbedder

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def detect_pose(results):
    if results.pose_landmarks is None:
        return None
    pose_landmarks = np.array([[lmk.x * 1, lmk.y * 1, lmk.z * 1]
                               for lmk in results.pose_landmarks.landmark], dtype=np.float32)

    pos_data = pd.DataFrame([])
    pose_encoder = FullBodyPoseEmbedder()
    pose_embedding = pose_encoder(pose_landmarks)
    pose_embedding = pose_embedding.flatten()
    n_features = len(pose_embedding)
    x_cols = [str(i) for i in range(n_features)]
    pos_data = pos_data.append({'data': pose_embedding}, ignore_index=True)
    pos_data[x_cols] = pd.DataFrame(pos_data['data'].tolist(), index=pos_data.index)

    with open('random_forest_classifier.pkl', 'rb') as fid:
        model_loaded = cPickle.load(fid)

    res = model_loaded.predict(pos_data[x_cols])

    pos_dict = {0: 'bed', 1: 'floor', 2: 'sitting', 3: 'standing'}
    return pos_dict[res.argmax()]
