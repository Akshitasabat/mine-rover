import time
# pyrefly: ignore [missing-import]
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from app.services.thermal_service import thermal_service

router = APIRouter(tags=["MLX90640 Thermal Camera & YOLOv6 Detection"])

NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}

def generate_thermal_stream():
    """Generates continuous MJPEG stream of reconstructed MLX90640 Matplotlib thermal frames."""
    while True:
        frame_bytes = thermal_service.get_latest_jpeg()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )
        time.sleep(0.05)  # ~20 FPS thermal stream delivery

@router.get("/thermal/video")
@router.get("/api/thermal/video")
def get_thermal_video_stream():
    """Returns live reconstructed MLX90640 Matplotlib thermal stream."""
    return StreamingResponse(
        generate_thermal_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=NO_CACHE_HEADERS,
    )

@router.get("/thermal/detections")
@router.get("/api/thermal/detections")
def get_thermal_detections():
    """Returns latest thermal YOLOv6 detection data (person_detected, confidence, bbox)."""
    return JSONResponse(
        content=thermal_service.get_latest_detection(),
        headers=NO_CACHE_HEADERS,
    )
