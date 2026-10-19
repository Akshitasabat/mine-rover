import os
import sys
import time
import threading
import cv2
import numpy as np
import torch

# Ensure YOLOv6 directory is in python path
YOLOV6_DIR = os.getenv("YOLOV6_DIR", r"C:\Users\akshita\Documents\SIH\Rover\YOLOv6")
if YOLOV6_DIR not in sys.path:
    sys.path.insert(0, YOLOV6_DIR)

try:
    from yolov6.layers.common import DetectBackend
    from yolov6.data.data_augment import letterbox
    from yolov6.utils.nms import non_max_suppression
    from yolov6.core.inferer import Inferer
    YOLOV6_AVAILABLE = True
except Exception as e:
    print(f"[YOLOv6 Service] Warning loading YOLOv6 modules: {e}")
    YOLOV6_AVAILABLE = False


class CameraYOLOv6Service:
    def __init__(self):
        # RTSP Configuration
        default_rtsp = "rtsp://admin:Yashsawant@192.168.0.104:5543/live/channel0"
        self.rtsp_url = os.getenv("CAMERA_RTSP_URL", default_rtsp)

        self.weights_path = os.getenv(
            "YOLOV6_WEIGHTS",
            os.path.join(YOLOV6_DIR, "yolov6s.pt")
        )
        self.img_size = 640
        self.conf_thres = 0.4
        self.iou_thres = 0.45

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.detector = None
        self.stride = 32

        # Threading Locks & Decoupled State
        self.frame_lock = threading.Lock()
        self.latest_raw_frame = None
        self.raw_frame_id = 0

        self.result_lock = threading.Lock()
        self.latest_jpeg = None
        self.latest_detections = {
            "count": 0,
            "persons": []
        }

        # FPS & Latency Metrics
        self.capture_fps = 0.0
        self.yolo_fps = 0.0
        self.inference_time_ms = 0.0

        self.running = False
        self.capture_thread = None
        self.inference_thread = None

        self._init_detector()

    def _init_detector(self):
        if not YOLOV6_AVAILABLE:
            print("[YOLOv6 Service] YOLOv6 modules unavailable.")
            return

        try:
            print(f"[YOLOv6 Service] Loading model weights from: {self.weights_path}")
            self.detector = DetectBackend(self.weights_path, device=self.device)
            self.stride = self.detector.stride
            self.detector.model.float()
            self.detector.model.eval()
            print("[YOLOv6 Service] YOLOv6-S detector successfully loaded and initialized!")
        except Exception as e:
            print(f"[YOLOv6 Service] Error loading YOLOv6 model: {e}")

    def start(self):
        if self.running:
            return
        self.running = True

        # Thread 1: Fast RTSP Frame Capture (No inference blocking)
        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.capture_thread.start()

        # Thread 2: Independent YOLOv6 Inference
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()

        print("[YOLOv6 Service] Decoupled RTSP capture & YOLO inference threads started.")

    def stop(self):
        self.running = False

    def _capture_worker(self):
        """Thread 1: Continuously reads frames from RTSP stream as fast as camera sends them.
        Always overwrites latest_raw_frame to prevent buffer accumulation and ensure ZERO LATENCY."""
        print(f"[RTSP Capture Thread] Connecting to CP-E23A RTSP Stream: {self.rtsp_url}")
        
        while self.running:
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                print("[RTSP Capture Thread] RTSP connection failed. Retrying in 2 seconds...")
                self._set_error_frame("CAM DISCONNECTED - RETRYING...")
                time.sleep(2)
                continue

            print("[RTSP Capture Thread] RTSP Stream connected! Continuous capture loop running...")

            frame_count = 0
            start_time = time.time()

            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[RTSP Capture Thread] Frame loss detected. Reconnecting RTSP...")
                    break

                # Overwrite single latest raw frame (NO QUEUE / NO WAITING FOR YOLO)
                with self.frame_lock:
                    self.latest_raw_frame = frame
                    self.raw_frame_id += 1

                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed >= 2.0:
                    self.capture_fps = round(frame_count / elapsed, 1)
                    frame_count = 0
                    start_time = time.time()

            cap.release()
            time.sleep(1)

    def _inference_worker(self):
        """Thread 2: Pulls the newest raw frame when ready, runs YOLOv6, and updates annotations."""
        print("[YOLO Inference Thread] Independent inference loop started.")
        last_processed_id = -1
        frame_count = 0
        start_time = time.time()

        while self.running:
            raw_frame = None
            current_id = -1

            with self.frame_lock:
                if self.latest_raw_frame is not None and self.raw_frame_id != last_processed_id:
                    raw_frame = self.latest_raw_frame.copy()
                    current_id = self.raw_frame_id

            if raw_frame is None:
                time.sleep(0.005)
                continue

            last_processed_id = current_id
            t0 = time.time()

            try:
                annotated_frame, det_data = self._process_frame(raw_frame)
                t1 = time.time()
                self.inference_time_ms = round((t1 - t0) * 1000, 1)

                _, jpeg_bytes = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

                with self.result_lock:
                    self.latest_jpeg = jpeg_bytes.tobytes()
                    self.latest_detections = det_data

                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed >= 2.0:
                    self.yolo_fps = round(frame_count / elapsed, 1)
                    print(f"[PERFORMANCE] Camera FPS: {self.capture_fps} | YOLO FPS: {self.yolo_fps} | Inference time: {self.inference_time_ms} ms")
                    frame_count = 0
                    start_time = time.time()

            except Exception as e:
                print(f"[YOLO Inference Thread] Error processing frame: {e}")
                time.sleep(0.01)

    def _process_frame(self, frame):
        if self.detector is None:
            cv2.putText(
                frame, "YOLOv6 Loading...", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2
            )
            return frame, {"count": 0, "persons": []}

        # Preprocess frame
        img = letterbox(frame, self.img_size, stride=self.stride)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_tensor = torch.from_numpy(np.ascontiguousarray(img)).float()
        img_tensor /= 255.0

        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor[None]

        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            pred_results = self.detector(img_tensor)
            det = non_max_suppression(
                pred_results,
                self.conf_thres,
                self.iou_thres,
                classes=[0],  # COCO Class 0 = Person
                agnostic=False,
                max_det=1000
            )[0]

        annotated = frame.copy()
        persons = []

        if len(det):
            det[:, :4] = Inferer.rescale(img_tensor.shape[2:], det[:, :4], frame.shape).round()
            for worker_idx, (*xyxy, conf, cls) in enumerate(det, start=1):
                if int(cls) == 0:
                    x1, y1, x2, y2 = [int(v.item()) for v in xyxy]
                    confidence = round(float(conf.item()), 2)
                    persons.append({
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2]
                    })

                    # Draw Bounding Box & Label
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 215, 255), 2)
                    label = f"Worker #{worker_idx} ({confidence * 100:.0f}%)"
                    
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(
                        annotated,
                        (x1, max(y1 - 22, 0)),
                        (x1 + t_size[0] + 6, max(y1, 22)),
                        (0, 215, 255),
                        -1
                    )
                    cv2.putText(
                        annotated,
                        label,
                        (x1 + 3, max(y1 - 6, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (15, 12, 10),
                        2
                    )

        # Performance & status text overlay on video
        perf_text = f"CAM: {self.capture_fps} FPS | YOLO: {self.yolo_fps} FPS | Latency: {self.inference_time_ms}ms"
        cv2.putText(
            annotated,
            perf_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 128),
            2
        )

        return annotated, {"count": len(persons), "persons": persons}

    def _set_error_frame(self, message):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            img, message, (100, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        _, jpeg = cv2.imencode('.jpg', img)
        with self.result_lock:
            self.latest_jpeg = jpeg.tobytes()
            self.latest_detections = {"count": 0, "persons": []}

    def get_latest_jpeg(self):
        with self.result_lock:
            if self.latest_jpeg is not None:
                return self.latest_jpeg
            
            # Default frame while initializing
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                img, "CP-E23A STREAM INITIALIZING...", (80, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2
            )
            _, jpeg = cv2.imencode('.jpg', img)
            return jpeg.tobytes()

    def get_latest_detections(self):
        with self.result_lock:
            return self.latest_detections


# Singleton instance
camera_service = CameraYOLOv6Service()
