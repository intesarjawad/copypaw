# Copyright 2018-2020 Open-MMLab.
# Licensed under the Apache License, Version 2.0.
# See LICENSE in this directory for the full licence.
# Modified for local CPU-only inference; see the project NOTICE.

import numpy as np
import onnxruntime as ort


class BaseTool:
    def __init__(self, onnx_model, model_input_size=None, mean=None, std=None):
        self.session = ort.InferenceSession(
            path_or_bytes=onnx_model, providers=["CPUExecutionProvider"]
        )
        if model_input_size is None:
            shape = self.session.get_inputs()[0].shape
            model_input_size = tuple(shape[2:4])
            if len(model_input_size) != 2 or any(
                not isinstance(size, int) or size <= 0 for size in model_input_size
            ):
                raise ValueError("The model must have fixed positive spatial dimensions")
        self.model_input_size = model_input_size
        self.mean = mean
        self.std = std

    def inference(self, img):
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        return self.session.run(None, {self.session.get_inputs()[0].name: img[None]})
