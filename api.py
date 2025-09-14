from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel, Field
from typing import Optional
from PIL import Image
from fastapi.responses import JSONResponse
import io

from analyzer import GuiScreenAnalyzer

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.get("/items/{item_id}", operation_id="get_item")
def get_item_endpoint(item_id: int, q: str | None = None):
    """
    Get an item by its ID.
    """
    return {"item_id": item_id, "q": q}

@app.post("/items/", operation_id="create_item")
def create_item_endpoint(item: Item):
    """
    Create an item.
    """
    return item

@app.post("/analyze_gui_screen", operation_id="analyze_gui_screen")
async def analyze_gui_screen_endpoint(
    image: UploadFile = File(...),
    box_threshold: float = Form(0.05, ge=0.01, le=1),
    iou_threshold: float = Form(0.1, ge=0.01, le=1),
    use_paddleocr: bool = Form(True),
    imgsz: int = Form(640, ge=640, le=1920),
):
    """
    Analyze a GUI screen from an image.
    """
    # Convert UploadFile to PIL.Image.Image
    image_content = await image.read()
    pil_image = Image.open(io.BytesIO(image_content))
    
    analyzer = GuiScreenAnalyzer(pil_image, box_threshold, iou_threshold, use_paddleocr, imgsz)
    label_image, result = analyzer.process()
    # save label_image to local
    # label_image.save("label_image.jpg")
    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)