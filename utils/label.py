import base64
import io
import time
from typing import List
import cv2
import numpy as np
import torch
from torchvision.transforms import ToPILImage
from PIL import Image
from torchvision.ops import box_convert
import supervision as sv
from ultralytics import YOLO
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM 
from utils.box_annotator import BoxAnnotator


class Label:
    def __init__(self, image: Image.Image, model=None, box_threshold=0.01, output_coord_in_ratio=False, ocr_bbox=None, text_scale=0.4, text_padding=5, draw_bbox_config=None, ocr_text=[], use_local_semantics=True, iou_threshold=0.9,prompt=None, scale_img=False, imgsz=None, batch_size=128):
        self.image = image
        self.box_threshold = box_threshold
        self.output_coord_in_ratio = output_coord_in_ratio
        self.ocr_bbox = ocr_bbox
        self.text_scale = text_scale
        self.text_padding = text_padding
        self.draw_bbox_config = draw_bbox_config
        self.ocr_text = ocr_text
        self.use_local_semantics = use_local_semantics
        self.iou_threshold = iou_threshold
        self.prompt = prompt
        self.scale_img = scale_img
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.yolo_model = YOLO('weights/icon_detect/model.pt')
        self.caption_model_name = 'florence2'
        self.caption_model_path = "weights/icon_caption_florence"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.caption_model_processor = self.get_caption_model_processor()


    def get_caption_model_processor(self):
        if self.caption_model_name == "blip2":
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            if self.device == 'cpu':
                model = Blip2ForConditionalGeneration.from_pretrained(
                self.caption_model_path, device_map=None, torch_dtype=torch.float32
            ) 
            else:
                model = Blip2ForConditionalGeneration.from_pretrained(
                self.caption_model_path, device_map=None, torch_dtype=torch.float16
            ).to(self.device)
        elif self.caption_model_name == "florence2":
            processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
            if self.device == 'cpu':
                model = AutoModelForCausalLM.from_pretrained(self.caption_model_path, torch_dtype=torch.float32, trust_remote_code=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(self.caption_model_path, torch_dtype=torch.float16, trust_remote_code=True).to(self.device)
        return {'model': model.to(self.device), 'processor': processor}



    def predict_yolo(self):
        """ Use huggingface model to replace the original model
        """
        # model = model['model']
        if self.scale_img:
            result = self.yolo_model.predict(
            source=self.image,
            conf=self.box_threshold,
            imgsz=self.imgsz,
            iou=self.iou_threshold, # default 0.7
            )
        else:
            result = self.yolo_model.predict(
            source=self.image,
            conf=self.box_threshold,
            iou=self.iou_threshold, # default 0.7
            )
        boxes = result[0].boxes.xyxy#.tolist() # in pixel space
        conf = result[0].boxes.conf
        phrases = [str(i) for i in range(len(boxes))]

        return boxes, conf, phrases

    @classmethod
    def int_box_area(cls, box, w, h):
        x1, y1, x2, y2 = box
        int_box = [int(x1*w), int(y1*h), int(x2*w), int(y2*h)]
        area = (int_box[2] - int_box[0]) * (int_box[3] - int_box[1])
        return area

    def remove_overlap_new(self, boxes, ocr_bbox=None):
        '''
        ocr_bbox format: [{'type': 'text', 'bbox':[x,y], 'interactivity':False, 'content':str }, ...]
        boxes format: [{'type': 'icon', 'bbox':[x,y], 'interactivity':True, 'content':None }, ...]

        '''
        assert ocr_bbox is None or isinstance(ocr_bbox, List)

        def box_area(box):
            return (box[2] - box[0]) * (box[3] - box[1])

        def intersection_area(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            return max(0, x2 - x1) * max(0, y2 - y1)

        def IoU(box1, box2):
            intersection = intersection_area(box1, box2)
            union = box_area(box1) + box_area(box2) - intersection + 1e-6
            if box_area(box1) > 0 and box_area(box2) > 0:
                ratio1 = intersection / box_area(box1)
                ratio2 = intersection / box_area(box2)
            else:
                ratio1, ratio2 = 0, 0
            return max(intersection / union, ratio1, ratio2)

        def is_inside(box1, box2):
            # return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]
            intersection = intersection_area(box1, box2)
            ratio1 = intersection / box_area(box1)
            return ratio1 > 0.80

        # boxes = boxes.tolist()
        filtered_boxes = []
        if ocr_bbox:
            filtered_boxes.extend(ocr_bbox)
        # print('ocr_bbox!!!', ocr_bbox)
        for i, box1_elem in enumerate(boxes):
            box1 = box1_elem['bbox']
            is_valid_box = True
            for j, box2_elem in enumerate(boxes):
                # keep the smaller box
                box2 = box2_elem['bbox']
                if i != j and IoU(box1, box2) > self.iou_threshold and box_area(box1) > box_area(box2):
                    is_valid_box = False
                    break
            if is_valid_box:
                if ocr_bbox:
                    # keep yolo boxes + prioritize ocr label
                    box_added = False
                    ocr_labels = ''
                    for box3_elem in ocr_bbox:
                        if not box_added:
                            box3 = box3_elem['bbox']
                            if is_inside(box3, box1): # ocr inside icon
                                # box_added = True
                                # delete the box3_elem from ocr_bbox
                                try:
                                    # gather all ocr labels
                                    ocr_labels += box3_elem['content'] + ' '
                                    filtered_boxes.remove(box3_elem)
                                except:
                                    continue
                                # break
                            elif is_inside(box1, box3): # icon inside ocr, don't added this icon box, no need to check other ocr bbox bc no overlap between ocr bbox, icon can only be in one ocr box
                                box_added = True
                                break
                            else:
                                continue
                    if not box_added:
                        if ocr_labels:
                            filtered_boxes.append({'type': 'icon', 'bbox': box1_elem['bbox'], 'interactivity': True, 'content': ocr_labels, 'source':'box_yolo_content_ocr'})
                        else:
                            filtered_boxes.append({'type': 'icon', 'bbox': box1_elem['bbox'], 'interactivity': True, 'content': None, 'source':'box_yolo_content_yolo'})
                else:
                    filtered_boxes.append(box1)
        return filtered_boxes # torch.tensor(filtered_boxes)


    @torch.inference_mode()
    def get_parsed_content_icon(self, filtered_boxes, starting_idx):
        # Number of samples per batch, --> 128 roughly takes 4 GB of GPU memory for florence v2 model
        to_pil = ToPILImage()
        if starting_idx:
            non_ocr_boxes = filtered_boxes[starting_idx:]
        else:
            non_ocr_boxes = filtered_boxes
        croped_pil_image = []
        for i, coord in enumerate(non_ocr_boxes):
            try:
                xmin, xmax = int(coord[0]*self.image.shape[1]), int(coord[2]*self.image.shape[1])
                ymin, ymax = int(coord[1]*self.image.shape[0]), int(coord[3]*self.image.shape[0])
                cropped_image = self.image[ymin:ymax, xmin:xmax, :]
                cropped_image = cv2.resize(cropped_image, (64, 64))
                croped_pil_image.append(to_pil(cropped_image))
            except:
                continue

        model, processor = self.caption_model_processor['model'], self.caption_model_processor['processor']
        if not self.prompt:
            if 'florence' in model.config.name_or_path:
                prompt = "<CAPTION>"
            else:
                prompt = "The image shows"
        
        generated_texts = []
        device = model.device
        for i in range(0, len(croped_pil_image), self.batch_size):
            batch = croped_pil_image[i:i+self.batch_size]
            if model.device.type == 'cuda':
                inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt", do_resize=False).to(device=device, dtype=torch.float16)
            else:
                inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt").to(device=device)
            if 'florence' in model.config.name_or_path:
                generated_ids = model.generate(input_ids=inputs["input_ids"],pixel_values=inputs["pixel_values"],max_new_tokens=20,num_beams=1, do_sample=False)
            else:
                generated_ids = model.generate(**inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True, num_return_sequences=1) # temperature=0.01, do_sample=True,
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            generated_text = [gen.strip() for gen in generated_text]
            generated_texts.extend(generated_text)
        
        return generated_texts

    def get_parsed_content_icon_phi3v(self, filtered_boxes):
        to_pil = ToPILImage()
        if self.ocr_bbox:
            non_ocr_boxes = filtered_boxes[len(self.ocr_bbox):]
        else:
            non_ocr_boxes = filtered_boxes
        croped_pil_image = []
        for i, coord in enumerate(non_ocr_boxes):
            xmin, xmax = int(coord[0]*self.image.shape[1]), int(coord[2]*self.image.shape[1])
            ymin, ymax = int(coord[1]*self.image.shape[0]), int(coord[3]*self.iamge.shape[0])
            cropped_image = self.image[ymin:ymax, xmin:xmax, :]
            croped_pil_image.append(to_pil(cropped_image))

        model, processor = self.caption_model_processor['model'], self.caption_model_processor['processor']
        device = model.device
        messages = [{"role": "user", "content": "<|image_1|>\ndescribe the icon in one sentence"}] 
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        batch_size = 5  # Number of samples per batch
        generated_texts = []

        for i in range(0, len(croped_pil_image), batch_size):
            images = croped_pil_image[i:i+batch_size]
            image_inputs = [processor.image_processor(x, return_tensors="pt") for x in images]
            inputs ={'input_ids': [], 'attention_mask': [], 'pixel_values': [], 'image_sizes': []}
            texts = [prompt] * len(images)
            for i, txt in enumerate(texts):
                input = processor._convert_images_texts_to_inputs(image_inputs[i], txt, return_tensors="pt")
                inputs['input_ids'].append(input['input_ids'])
                inputs['attention_mask'].append(input['attention_mask'])
                inputs['pixel_values'].append(input['pixel_values'])
                inputs['image_sizes'].append(input['image_sizes'])
            max_len = max([x.shape[1] for x in inputs['input_ids']])
            for i, v in enumerate(inputs['input_ids']):
                inputs['input_ids'][i] = torch.cat([processor.tokenizer.pad_token_id * torch.ones(1, max_len - v.shape[1], dtype=torch.long), v], dim=1)
                inputs['attention_mask'][i] = torch.cat([torch.zeros(1, max_len - v.shape[1], dtype=torch.long), inputs['attention_mask'][i]], dim=1)
            inputs_cat = {k: torch.concatenate(v).to(device) for k, v in inputs.items()}

            generation_args = { 
                "max_new_tokens": 25, 
                "temperature": 0.01, 
                "do_sample": False, 
            } 
            generate_ids = model.generate(**inputs_cat, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 
            # # remove input tokens 
            generate_ids = generate_ids[:, inputs_cat['input_ids'].shape[1]:]
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            response = [res.strip('\n').strip() for res in response]
            generated_texts.extend(response)

        return generated_texts

    def annotate(self, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str], text_scale: float, 
                text_padding=5, text_thickness=2, thickness=3) -> np.ndarray:
        """    
        This function annotates an image with bounding boxes and labels.

        Parameters:
        image_source (np.ndarray): The source image to be annotated.
        boxes (torch.Tensor): A tensor containing bounding box coordinates. in cxcywh format, pixel scale
        logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
        phrases (List[str]): A list of labels for each bounding box.
        text_scale (float): The scale of the text to be displayed. 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web

        Returns:
        np.ndarray: The annotated image.
        """
        h, w, _ = self.image.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
        detections = sv.Detections(xyxy=xyxy)

        labels = [f"{phrase}" for phrase in range(boxes.shape[0])]

        box_annotator = BoxAnnotator(text_scale=text_scale, text_padding=text_padding,text_thickness=text_thickness,thickness=thickness) # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
        annotated_frame = self.image.copy()
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels, image_size=(w,h))

        label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
        return annotated_frame, label_coordinates


    def process(self):
        """
        Process either an image path or Image object
        """
        self.image = self.image.convert("RGB") # for CLIP
        w, h = self.image.size
        if not self.imgsz:
            self.imgsz = (h, w)
        # print('image size:', w, h)
        xyxy, logits, phrases = self.predict_yolo()
        xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
        self.image = np.asarray(self.image)
        phrases = [str(i) for i in range(len(phrases))]

        # annotate the image with labels
        if self.ocr_bbox:
            self.ocr_bbox = torch.tensor(self.ocr_bbox) / torch.Tensor([w, h, w, h])
            self.ocr_bbox=self.ocr_bbox.tolist()
        else:
            print('no ocr bbox!!!')
            self.ocr_bbox = None

        ocr_bbox_elem = [{'type': 'text', 'bbox':box, 'interactivity':False, 'content':txt, 'source': 'box_ocr_content_ocr'} for box, txt in zip(self.ocr_bbox, self.ocr_text) if self.int_box_area(box, w, h) > 0] 
        xyxy_elem = [{'type': 'icon', 'bbox':box, 'interactivity':True, 'content':None} for box in xyxy.tolist() if self.int_box_area(box, w, h) > 0]
        filtered_boxes = self.remove_overlap_new(xyxy_elem,  ocr_bbox_elem)
        
        # sort the filtered_boxes so that the one with 'content': None is at the end, and get the index of the first 'content': None
        filtered_boxes_elem = sorted(filtered_boxes, key=lambda x: x['content'] is None)
        # get the index of the first 'content': None
        starting_idx = next((i for i, box in enumerate(filtered_boxes_elem) if box['content'] is None), -1)
        filtered_boxes = torch.tensor([box['bbox'] for box in filtered_boxes_elem])
        print('len(filtered_boxes):', len(filtered_boxes), starting_idx)

        # get parsed icon local semantics
        time1 = time.time()
        if self.use_local_semantics:
            caption_model = self.caption_model_processor['model']
            if 'phi3_v' in caption_model.config.model_type: 
                parsed_content_icon = self.get_parsed_content_icon_phi3v(filtered_boxes)
            else:
                parsed_content_icon = self.get_parsed_content_icon(filtered_boxes, starting_idx)
            self.ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(self.ocr_text)]
            icon_start = len(self.ocr_text)
            parsed_content_icon_ls = []
            # fill the filtered_boxes_elem None content with parsed_content_icon in order
            for i, box in enumerate(filtered_boxes_elem):
                if box['content'] is None:
                    box['content'] = parsed_content_icon.pop(0)
            for i, txt in enumerate(parsed_content_icon):
                parsed_content_icon_ls.append(f"Icon Box ID {str(i+icon_start)}: {txt}")
        else:
            self.ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(self.ocr_text)]
        print('time to get parsed content:', time.time()-time1)

        filtered_boxes = box_convert(boxes=filtered_boxes, in_fmt="xyxy", out_fmt="cxcywh")

        phrases = [i for i in range(len(filtered_boxes))]
        
        # draw boxes
        if self.draw_bbox_config:
            annotated_frame, label_coordinates = self.annotate(boxes=filtered_boxes, logits=logits, phrases=phrases, **self.draw_bbox_config)
        else:
            annotated_frame, label_coordinates = self.annotate(boxes=filtered_boxes, logits=logits, phrases=phrases, text_scale=self.text_scale, text_padding=self.text_padding)
        
        pil_img = Image.fromarray(annotated_frame)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('ascii')
        if self.output_coord_in_ratio:
            label_coordinates = {k: [v[0]/w, v[1]/h, v[2]/w, v[3]/h] for k, v in label_coordinates.items()}
            assert w == annotated_frame.shape[1] and h == annotated_frame.shape[0]

        return encoded_image, label_coordinates, filtered_boxes_elem
