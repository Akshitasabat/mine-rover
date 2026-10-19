import time
# pyrefly: ignore [missing-import]
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from app.services.yolov6_service import camera_service

router = APIRouter(tags=["Camera & YOLOv6 Detection"])

NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0"
}

def generate_video_stream():
    """Generates continuous multipart MJPEG stream from latest YOLOv6 annotated frame."""
    while True:
        frame_bytes = camera_service.get_latest_jpeg()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )
        time.sleep(0.03)  # ~33 FPS stream delivery

@router.get("/video")
@router.get("/api/video")
def get_video_stream():
    """Returns live annotated RTSP camera feed with zero caching."""
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=NO_CACHE_HEADERS
    )

@router.get("/detections", headers=NO_CACHE_HEADERS)
@router.get("/api/detections", headers=NO_CACHE_HEADERS)
def get_detections():
    """Returns latest YOLOv6 detection data without triggering new inference."""
    return JSONResponse(
        content=camera_service.get_latest_detections(),
        headers=NO_CACHE_HEADERS
    )
