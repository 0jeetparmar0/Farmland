

import cv2
import numpy as np
import os
from timeit import default_timer as timer

class FarmlandEdgeDetector:
    def __init__(self, model_path, weights_path, use_cuda=False):
        self.net = cv2.dnn.readNetFromCaffe(model_path, weights_path)
        if use_cuda:
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                dummy = cv2.dnn.blobFromImage(np.zeros((32, 32, 3), dtype=np.uint8))
                self.net.setInput(dummy)
                self.net.forward()
                print("[INFO] CUDA backend is active.")
            except Exception as e:
                print(f"[WARNING] CUDA backend failed: {e}. Falling back to CPU.")
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("[INFO] CPU backend is active.")

    def enhance_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def detect_edges_from_array(self, image, threshold=0.5):
        image = self.enhance_contrast(image)
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0,
            size=(width, height),
            mean=(104.00698793, 116.66876762, 122.67891434),
            swapRB=False,
            crop=False
        )
        self.net.setInput(blob)
        hed = self.net.forward()
        hed = hed[0, 0]
        hed = cv2.resize(hed, (width, height), interpolation=cv2.INTER_LINEAR)
        hed = (255 * hed).astype("uint8")
        _, binary_mask = cv2.threshold(hed, int(threshold * 255), 255, cv2.THRESH_BINARY)
        return binary_mask, None

    def detect_edges_in_tiles(self, image_path, output_path, threshold=0.5, tile_size=1024, overlap=64):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)

        step = tile_size - overlap
        for y in range(0, h, step):
            for x in range(0, w, step):
                x_end = min(x + tile_size, w)
                y_end = min(y + tile_size, h)
                tile = image[y:y_end, x:x_end]
                processed_tile, _ = self.detect_edges_from_array(tile, threshold)
                ph, pw = processed_tile.shape

                mask[y:y+ph, x:x+pw] += processed_tile.astype(np.float32)
                weight[y:y+ph, x:x+pw] += 1.0

        weight[weight == 0] = 1.0
        final_mask = (mask / weight).astype(np.uint8)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, final_mask)
        print(f"[INFO] Saved tiled mask to {output_path}")
        return final_mask

if __name__ == "__main__":
    detector = FarmlandEdgeDetector(
        model_path='/home/jazzy/sem-farm/Hed/models/deploy.prototxt',
        weights_path='/home/jazzy/sem-farm/Hed/models/hed_pretrained_bsds.caffemodel',
        use_cuda=True
    )

    detector.detect_edges_in_tiles(
        image_path='/home/jazzy/sem-farm/data/final_farmland_jpg.jpg',
        output_path='/home/jazzy/sem-farm/Hed/output_hed/farmland_mask_tiled.png',
        threshold=0.7,
        tile_size=1024,
        overlap=64
    )
