import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import sys

workdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(workdir)

from scripts.pose_encoder import FullBodyPoseEmbedder

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
img = cv2.imread('../images.jpg')


def extract_kinematic_model(img):
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=True) as pose:
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return results


def segment_person(img, results):
    annotated_image = img.copy()
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    segm = np.repeat(results.segmentation_mask[..., np.newaxis], 3, axis=2)
    annotated_image = annotated_image * segm

    return annotated_image


def inpaint_person(img, results):
    annotated_image = cv2.inpaint(img, np.array(results.segmentation_mask, 'uint8'), 3, cv2.INPAINT_TELEA)
    return annotated_image


def extract_kinematic_features(img):
    kinematic_model = extract_kinematic_model(img)
    pose_landmarks = np.array([[lmk.x * 1, lmk.y * 1, lmk.z * 1]
                               for lmk in kinematic_model.pose_landmarks.landmark], dtype=np.float32)
    pose_encoder = FullBodyPoseEmbedder()
    pose_embedding = pose_encoder(pose_landmarks)
    pose_embedding = pose_embedding.flatten()
    return pose_embedding


# features = extract_kinematic_features(img)
# results = extract_kinematic_model(img)
# img = segment_person(img, results)
# cv2.imwrite('images_2.jpeg', img)

features = extract_kinematic_features(img)
