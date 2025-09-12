from fastapi import FastAPI, File, UploadFile, Form
from fastapi_mcp import FastApiMCP
from pydantic import BaseModel, Field
from typing import Optional

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
def analyze_gui_screen_endpoint(
    image: UploadFile = File(...),
    box_threshold: float = Form(0.05, ge=0.01, le=1),
    iou_threshold: float = Form(0.1, ge=0.01, le=1),
    use_paddleocr: bool = Form(True),
    imgsz: int = Form(640, ge=640, le=1920),
):
    """
    Analyze a GUI screen from an image.
    """
    return {
        "filename": image.filename,
        "content_type": image.content_type,
        "box_threshold": box_threshold,
        "iou_threshold": iou_threshold,
        "use_paddleocr": use_paddleocr,
        "imgsz": imgsz,
    }

# Create the MCP server from the FastAPI app
mcp = FastApiMCP(app)
# Mount the MCP server at /mcp
mcp.mount()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
