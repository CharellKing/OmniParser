from fastapi import FastAPI, File, UploadFile, Form
from fastapi_mcp import FastApiMCP
from pydantic import BaseModel, Field
from typing import Optional
from PIL import Image
from fastapi.responses import JSONResponse
import io

from analyzer import GuiScreenAnalyzer
from fastapi.staticfiles import StaticFiles

app = FastAPI()
mcp = FastApiMCP(app)
mcp.mount()


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


# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)