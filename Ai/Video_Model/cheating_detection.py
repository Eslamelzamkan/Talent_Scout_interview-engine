"""
Ai/Video_Model/cheating_detection.py

Detects gaze cheating in interview videos using MediaPipe FaceMesh
and iris landmark tracking.
"""

import cv2
import asyncio
import logging
import mediapipe as mp

logger = logging.getLogger(__name__)


class CheatingDetection:
    """
    Detects if a candidate was looking away from the camera for more than 3
    consecutive seconds during the interview.

    Returns:
        1  — gaze was centered (no cheating detected)
        0  — cheating detected (candidate looked away for >= 3 seconds)
       -1  — video could not be opened
    """

    _LEFT_EYE   = [33, 133]
    _RIGHT_EYE  = [362, 263]
    _LEFT_IRIS  = 468
    _RIGHT_IRIS = 473

    _NON_CENTER_THRESHOLD = 3.0
    _CENTER_CONFIRM_TIME  = 0.5
    _GAZE_CENTER_LOW      = 0.35
    _GAZE_CENTER_HIGH     = 0.65

    async def detect_gaze_cheating_async(self, video_path: str) -> int:
        """Analyse gaze frame-by-frame. Returns 0 (cheating) or 1 (ok)."""
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            static_image_mode=False,
            max_num_faces=1,
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return -1

        fps         = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        non_center_time = 0.0
        center_counter  = 0.0
        cheating        = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for lm in results.multi_face_landmarks:
                    l_iris_x = int(lm.landmark[self._LEFT_IRIS].x  * frame_width)
                    r_iris_x = int(lm.landmark[self._RIGHT_IRIS].x * frame_width)
                    l_left   = int(lm.landmark[self._LEFT_EYE[0]].x  * frame_width)
                    l_right  = int(lm.landmark[self._LEFT_EYE[1]].x  * frame_width)
                    r_left   = int(lm.landmark[self._RIGHT_EYE[0]].x * frame_width)
                    r_right  = int(lm.landmark[self._RIGHT_EYE[1]].x * frame_width)

                    if self._is_centered(l_iris_x, l_left, l_right) and \
                       self._is_centered(r_iris_x, r_left, r_right):
                        center_counter  += 1 / fps
                        if center_counter >= self._CENTER_CONFIRM_TIME:
                            non_center_time = 0.0
                    else:
                        center_counter   = 0.0
                        non_center_time += 1 / fps
                        if non_center_time >= self._NON_CENTER_THRESHOLD:
                            cheating = True
                            break
            else:
                center_counter   = 0.0
                non_center_time += 1 / fps
                if non_center_time >= self._NON_CENTER_THRESHOLD:
                    cheating = True
                    break

            await asyncio.sleep(0)

        cap.release()
        face_mesh.close()

        logger.info(f"Cheating detection result for {video_path}: cheating={cheating}")
        return 0 if cheating else 1

    def _is_centered(self, iris_x: int, eye_left: int, eye_right: int) -> bool:
        """Return True if the iris is within the centered gaze zone."""
        eye_width = eye_right - eye_left
        if eye_width == 0:
            return False
        rel = (iris_x - eye_left) / eye_width
        return self._GAZE_CENTER_LOW < rel < self._GAZE_CENTER_HIGH
