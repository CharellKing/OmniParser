from typing import Union
from paddleocr import PaddleOCR
import easyocr
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt



class OCR:
    paddle_ocr = PaddleOCR(
        lang='ch,en',
        use_angle_cls=False,
        use_gpu=False,  # using cuda will conflict with pytorch in the same process
        show_log=False,
        max_batch_size=1024,
        use_dilation=True,  # improves accuracy
        det_db_score_mode='slow',  # improves accuracy
        rec_batch_num=1024)

    easy_ocr = easyocr.Reader(['ch_sim', 'en'])

    def __init__(self, image_source: Union[str, Image.Image], display_img = True, output_bb_format='xywh', goal_filtering=None, easyocr_args=None, use_paddleocr=False):
        if isinstance(image_source, str):
            self.image = Image.open(image_source)
        else:
            self.image = image_source

        self.display_img = display_img
        self.output_bb_format = output_bb_format
        self.goal_filtering = goal_filtering
        self.easyocr_args = easyocr_args
        self.use_paddleocr = use_paddleocr

    @classmethod
    def get_xywh(cls, input):
        x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
        x, y, w, h = int(x), int(y), int(w), int(h)
        return x, y, w, h

    @classmethod
    def get_xyxy(cls, input):
        x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
        x, y, xp, yp = int(x), int(y), int(xp), int(yp)
        return x, y, xp, yp

    def check_ocr_box(self):
        if isinstance(self.image, str):
            self.image = Image.open(self.image)

        if self.image.mode == 'RGBA':
            # Convert RGBA to RGB to avoid alpha channel issues
            self.image = self.image.convert('RGB')
        image_np = np.array(self.image)
        w, h = self.image.size
        if self.use_paddleocr:
            if self.easyocr_args is None:
                text_threshold = 0.5
            else:
                text_threshold = self.easyocr_args['text_threshold']
            result = self.paddle_ocr.ocr(image_np, cls=False)[0]
            coord = [item[0] for item in result if item[1][1] > text_threshold]
            text = [item[1][0] for item in result if item[1][1] > text_threshold]
        else:  # EasyOCR
            if self.easyocr_args is None:
                self.easyocr_args = {}
            result = self.easy_ocr.readtext(image_np, **self.easyocr_args)
            coord = [item[0] for item in result]
            text = [item[1] for item in result]
        
        if self.display_img:
            opencv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            bb = []
            for item in coord:
                x, y, a, b = self.get_xywh(item)
                bb.append((x, y, a, b))
                cv2.rectangle(opencv_img, (x, y), (x+a, y+b), (0, 255, 0), 2)
            #  matplotlib expects RGB
            plt.imshow(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))
        else:
            if self.output_bb_format == 'xywh':
                bb = [self.get_xywh(item) for item in coord]
            elif self.output_bb_format == 'xyxy':
                bb = [self.get_xyxy(item) for item in coord]
        return (text, bb), self.goal_filtering