import base64
import io
from typing import Union
from PIL import Image
from paddleocr import PaddleOCR
from utils.label import Label
from utils.ocr import OCR


class GuiScreenAnalyzer:
    def __init__(self, image_source: Union[Image.Image, str], box_threshold: float, iou_threshold: float, use_paddleocr: bool, image_size: int):
        if isinstance(image_source, str):
            self.image = Image.open(image_source)
        else:
            self.image = image_source
    
        self.box_threshold = box_threshold
        self.iou_threshold = iou_threshold
        self.use_paddleocr = use_paddleocr
        self.image_size = image_size

        self.box_overlay_ratio = self.image.size[0] / 3200
        self.draw_bbox_config = {
            'text_scale': 0.8 * self.box_overlay_ratio,
            'text_thickness': max(int(2 * self.box_overlay_ratio), 1),
            'text_padding': max(int(3 * self.box_overlay_ratio), 1),
            'thickness': max(int(3 * self.box_overlay_ratio), 1),
        }

    def process(self):
        ocr = OCR(self.image, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=self.use_paddleocr)
        ocr_bbox_rslt, is_goal_filtered = ocr.check_ocr_box()

        text, ocr_bbox = ocr_bbox_rslt
        label = Label(self.image, box_threshold=self.box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=self.draw_bbox_config, caption_model_processor=self.caption_model_processor, ocr_text=text,iou_threshold=self.iou_threshold, imgsz=self.imgsz,)
        dino_labled_img, label_coordinates, parsed_content_list = label.process()
        image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
        print('finish processing')
        parsed_content_list = '\n'.join([f'icon {i}: ' + str(v) for i,v in enumerate(parsed_content_list)])
        return image, str(parsed_content_list)