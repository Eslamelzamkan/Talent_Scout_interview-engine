"""
Ai/Video_Model/personality_traits.py

Extracts Big-Five personality trait scores from interview video using:
  1. MTCNN face detection
  2. X3D video backbone with 5 regression heads

The X3D model architecture is defined inline (was previously in x3d_model.py)
to keep the module self-contained.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import av
import os
from torchvision import transforms
from facenet_pytorch import MTCNN

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# X3D Model Architecture (merged from x3d_model.py)
# ─────────────────────────────────────────────────────────────────────────────

class x3d_model(nn.Module):
    """
    X3D feature extractor + 5 independent regression heads.

    Input:  (B, C, T, H, W) video tensor
    Output: (B, 5) trait scores

    NOTE: Attribute names prod1–prod5 match the state_dict keys in
    X3D_Third_CheckPoint.pth.  Do NOT rename them.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.feature_extractor = torch.hub.load(
            "facebookresearch/pytorchvideo", model_name, pretrained=True
        )

        self.prod1 = nn.Sequential(
            nn.Linear(400, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        self.prod2 = nn.Sequential(
            nn.Linear(400, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        self.prod3 = nn.Sequential(
            nn.Linear(400, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        self.prod4 = nn.Sequential(
            nn.Linear(400, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        self.prod5 = nn.Sequential(
            nn.Linear(400, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(frames)
        features = features.flatten(start_dim=1)
        return torch.cat([
            self.prod1(features),
            self.prod2(features),
            self.prod3(features),
            self.prod4(features),
            self.prod5(features),
        ], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Video Personality Traits Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class Video_PersonalityTraits:
    """
    Predicts Big-Five personality trait scores from a candidate's video.

    Pipeline:
      extract_frames() → _extract_faces() → X3D model → trait scores [AGR, CONN, EXT, NEU, OPN]
    """

    def __init__(self, frame_count: int = 30, image_size: int = 160, margin: int = 40):
        self.device      = "cuda" if torch.cuda.is_available() else "cpu"
        self.frame_count = frame_count
        self.image_size  = image_size
        self.is_fallback = False
        self.model = None
        self.face_detector = None

        weights_path = os.path.join(os.path.dirname(__file__), "X3D_Third_CheckPoint.pth")
        if not os.path.exists(weights_path):
            self.is_fallback = True
            logger.warning(
                "Video personality checkpoint not found at %s. Using text-only trait fallback.",
                weights_path,
            )
            return

        self.face_detector = MTCNN(
            image_size=image_size,
            margin=margin,
            post_process=False,
            select_largest=True,
            device=self.device,
        )

        self.model = x3d_model("x3d_s")
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device)
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def process_new_video(self, video_path: str) -> list:
        """
        Run the full pipeline on a single video.

        Returns a flat list of 5 trait scores: [AGR, CONN, EXT, NEU, OPN]
        """
        if self.is_fallback or self.model is None:
            return [0.5, 0.5, 0.5, 0.5, 0.5]

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        frames = self.extract_frames(video_path)
        frames = frames.unsqueeze(0).to(self.device)
        frames = frames.permute(0, 2, 1, 3, 4)

        with torch.no_grad():
            predictions = self.model(frames)

        return predictions.cpu().numpy().tolist()[0]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def extract_frames(self, video_path: str) -> torch.Tensor:
        """Extract up to 60 evenly-spaced frames from the video, then detect faces."""
        try:
            container = av.open(video_path)
        except Exception as e:
            logger.error(f"Could not open video {video_path}: {e}")
            return torch.zeros(self.frame_count, 3, self.image_size, self.image_size)

        frames = []
        total_frames = container.streams.video[0].frames

        if total_frames and total_frames > 0:
            indices = set(np.linspace(0, total_frames - 1, 60, dtype=int))
            for i, frame in enumerate(container.decode(video=0)):
                if i in indices:
                    img = frame.to_ndarray(format="rgb24")
                    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                    frames.append(img)
                if len(frames) >= 60:
                    break
        else:
            temp = []
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format="rgb24")
                img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                temp.append(img)

            if temp:
                indices = np.linspace(0, len(temp) - 1, min(60, len(temp)), dtype=int)
                frames = [temp[i] for i in indices]

        if not frames:
            logger.warning(f"No frames extracted from {video_path}; returning black frames.")
            return torch.zeros(self.frame_count, 3, self.image_size, self.image_size)

        return self._extract_faces(frames)

    def _extract_faces(self, frames: list) -> torch.Tensor:
        """Detect and crop faces from raw frames."""
        face_tensors = []
        for frame in frames:
            img  = transforms.ToPILImage()(frame)
            face = self.face_detector(img)
            if face is not None:
                face_tensors.append(face)
            if len(face_tensors) >= self.frame_count:
                break
        return self._pad_frames(face_tensors)

    def _pad_frames(self, frames: list) -> torch.Tensor:
        """Ensure exactly `frame_count` frames by repeating/truncating."""
        if not frames:
            return torch.zeros(self.frame_count, 3, self.image_size, self.image_size)

        n = len(frames)
        if n < self.frame_count:
            indices = np.linspace(0, n - 1, self.frame_count, dtype=int)
            frames  = [frames[i] for i in indices]
        else:
            frames = frames[: self.frame_count]

        return torch.stack(frames)
