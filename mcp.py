from mcp.server.fastmcp import FastMCP
from analyzer import GuiScreenAnalyzer
from PIL import Image


mcp = FastMCP("omniparser")
USER_AGENT = "omniparser-app/1.0"


@mcp.tool()
def process(image: File, box_threshold=0.05, iou_threshold=0.1, use_paddleocr=True, imgsz=640) -> dict:
    """
    MCP服务调用，处理图像并返回标注结果。

    Args:
        image (File): 输入的PIL图像。
        box_threshold (float): 框过滤阈值。
        iou_threshold (float): IoU过滤阈值。
        use_paddleocr (bool): 是否使用PaddleOCR。
        imgsz (int): 图像尺寸。

    Returns:
        tuple: 标注后的图像和解析结果。
    """
    image_content = image.read()
    pil_image = Image.open(io.BytesIO(image_content))
    
    analyzer = GuiScreenAnalyzer(pil_image, box_threshold, iou_threshold, use_paddleocr, imgsz)
    label_image, result = analyzer.process()
    return result

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
