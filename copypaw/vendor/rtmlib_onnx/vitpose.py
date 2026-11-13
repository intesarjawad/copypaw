# Copyright 2018-2020 Open-MMLab.
# Licensed under the Apache License, Version 2.0.
# See LICENSE in this directory for the full licence.
# Modified for local CPU-only inference; see the project NOTICE.

from typing import List, Tuple

import numpy as np

from .base import BaseTool
from .pose_postprocess import post_dark_udp
from .pose_preprocess import bbox_xyxy2cs, top_down_affine


class ViTPose(BaseTool):
    def __init__(
        self,
        onnx_model: str,
        model_input_size: tuple = (192, 256),
        mean: tuple = (123.675, 116.28, 103.53),
        std: tuple = (58.395, 57.12, 57.375),
    ):
        super().__init__(onnx_model, model_input_size, mean, std)

    def __call__(self, image: np.ndarray, bboxes=None):
        if bboxes is None or len(bboxes) == 0:
            bboxes = [[0, 0, image.shape[1], image.shape[0]]]

        keypoints, scores = [], []
        for bbox in bboxes:
            img, center, scale = self.preprocess(image, bbox)
            outputs = self.inference(img)
            kpts, score = self.postprocess(outputs, center, scale)

            keypoints.append(kpts)
            scores.append(score)

        keypoints = np.concatenate(keypoints, axis=0)
        scores = np.concatenate(scores, axis=0)

        return keypoints, scores

    def preprocess(self, img: np.ndarray, bbox: list):
        """Crop and normalize an RGB image for ViTPose inference.

        Args:
            img (np.ndarray): RGB image in shape (H, W, 3).
            bbox (list):  xyxy-format bounding box of target.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Bounding-box center in source pixels.
            - scale (np.ndarray): Padded crop extent in source pixels.
        """
        bbox = np.array(bbox)

        # get center and scale
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)

        # do affine transformation
        resized_img, scale = top_down_affine(self.model_input_size, scale, center, img)
        # normalize image
        if self.mean is not None:
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)
            resized_img = (resized_img - self.mean) / self.std

        return resized_img, center, scale

    def postprocess(
        self,
        outputs: List[np.ndarray],
        center: Tuple[int, int],
        scale: Tuple[int, int],
        score_threshold: float = 0.0,
        dark_kernel: int = 11,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess for ViTPose model output.

        Args:
            outputs (list[np.ndarray]): Output heatmaps of ViTPose model.
            center (tuple): Center of bbox in shape (x, y).
            scale (tuple): Scale of bbox in shape (w, h).
            score_threshold (float): Threshold to filter out low score keypoints.
            dark_kernel (int): Kernel size for DARK post-processing.

        Returns:
            tuple:
            - keypoints (np.ndarray): Rescaled keypoints.
            - scores (np.ndarray): Model predict scores.
        """
        # extract heatmaps
        heatmaps = outputs[0]
        N, K, H, W = heatmaps.shape

        # get initial keypoints and scores from heatmaps
        heatmaps_reshaped = heatmaps.reshape((N, K, -1))
        idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
        scores = np.max(heatmaps_reshaped, 2).reshape((N, K, 1))

        # convert flattened indices to 2D coordinates
        keypoints = np.tile(idx, (1, 1, 2)).astype(np.float32)
        keypoints[..., 0] = keypoints[..., 0] % W
        keypoints[..., 1] = keypoints[..., 1] // W

        # filter low-confidence keypoints
        keypoints = np.where(np.tile(scores, (1, 1, 2)) > score_threshold, keypoints, -1)

        # apply DARK post-processing for sub-pixel accuracy
        keypoints = post_dark_udp(keypoints, heatmaps, kernel=dark_kernel)

        # APT-36K uses ap10k_train_pipeline: ordinary TopDownAffine and MSRA
        # targets, including in all small/base/large multi-dataset configs:
        # https://github.com/ViTAE-Transformer/ViTPose/blob/main/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose%2B_small_coco%2Baic%2Bmpii%2Bap10k%2Bapt36k%2Bwholebody_256x192_udp.py#L297-L320
        # The export selects the APT-36K head without changing its geometry:
        # https://github.com/JunkyByte/easy_ViTPose/blob/main/model_split.py#L67-L93
        # https://github.com/JunkyByte/easy_ViTPose/blob/main/export.py#L18-L53
        # https://github.com/JunkyByte/easy_ViTPose/blob/main/easy_ViTPose/configs/ViTPose_apt36k.py
        # https://github.com/JunkyByte/easy_ViTPose/blob/main/easy_ViTPose/configs/ViTPose_common.py#L84-L91
        # Its generic use_udp test setting conflicts with animal training;
        # decode with the trained W/H stride, matching the ordinary crop.
        keypoints = keypoints / np.array([W, H]) * scale
        keypoints = keypoints + center - scale / 2

        scores = np.squeeze(scores, axis=2)

        return keypoints, scores
