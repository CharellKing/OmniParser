from fastapi import FastAPI, File, UploadFile, Form
from fastapi_mcp import FastApiMCP
from pydantic import BaseModel, Field
from typing import Optional
from PIL import Image
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from analyzer import GuiScreenAnalyzer
from fastapi.staticfiles import StaticFiles

import io
import os
import time


app = FastAPI()
mcp = FastApiMCP(app)
mcp.mount()

static_dir = f"{os.path.dirname(__file__)}/static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

class BoxBoundary(BaseModel):
    left: float = Field(..., description="Left coordinate of the bounding box")
    top: float = Field(..., description="Top coordinate of the bounding box")
    right: float = Field(..., description="Right coordinate of the bounding box")
    bottom: float = Field(..., description="Bottom coordinate of the bounding box")

class GUIElementBox(BaseModel):
    type: str = Field(..., description="Type of the element")
    bbox: BoxBoundary = Field(..., description="Bounding box coordinates in xyxy format")
    interactivity: bool = Field(..., description="Whether the element is interactive")
    content: str = Field(..., description="Content of the element")
    source: str = Field(..., description="Source of the element")

class GUIElementResponse(BaseModel):
    gui_element_box_list: list[GUIElementBox] = Field(..., description="List of GUI element boxes")
    image_url: str = Field(..., description="URL of the analyzed image")

@app.post("/analyze_gui_screen", operation_id="analyze_gui_screen")
async def analyze_gui_screen_endpoint(
    image: UploadFile = File(...),
    box_threshold: float = Form(0.05, ge=0.01, le=1),
    iou_threshold: float = Form(0.1, ge=0.01, le=1),
    use_paddleocr: bool = Form(True),
    imgsz: int = Form(640, ge=640, le=1920),
) -> GUIElementResponse:
    """
    Analyze a GUI screen from an image.
    """
    # Convert UploadFile to PIL.Image.Image
    image_content = await image.read()
    pil_image = Image.open(io.BytesIO(image_content))
    
    analyzer = GuiScreenAnalyzer(pil_image, box_threshold, iou_threshold, use_paddleocr, imgsz)
    label_image, parsed_content_list = analyzer.process()

    # save label_image to static_dir, name with nano timestamp
    label_image_path = f"{static_dir}/{time.time_ns()}.jpg"
    label_image.save(label_image_path)

    # convert parsed_content_list to GUIElementBox
    gui_element_box_list = []
    for parsed_content in parsed_content_list:
        bbox = parsed_content['bbox']
        gui_element_box_list.append(GUIElementBox(
            type=parsed_content['type'],
            bbox=BoxBoundary(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3]),
            interactivity=parsed_content['interactivity'],
            content=parsed_content['content'],
            source=parsed_content['source'],
        ))

    return GUIElementResponse(
        gui_element_box_list=gui_element_box_list,
        image_url=label_image_path,
    )


# Mount static files directory
app.mount(static_dir, StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)