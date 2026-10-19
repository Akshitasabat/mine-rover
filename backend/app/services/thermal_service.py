import os
import sys
import time
import json
import ssl
import io
import threading
import numpy as np
import cv2
import torch

import matplotlib
matplotlib.use('Agg')  # Server thread-safe non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import paho.mqtt.client as mqtt

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
    print(f"[Thermal Service] Warning loading YOLOv6 modules: {e}")
    YOLOV6_AVAILABLE = False


class MLX90640ThermalService:
    def __init__(self):
        # MQTT Config
        self.mqtt_host = os.getenv("MQTT_HOST", "76a2a993f5bb49efb3e0f723ddad5d8a.s1.eu.hivemq.cloud")
        self.mqtt_port = int(os.getenv("MQTT_PORT", "8883"))
        self.mqtt_username = os.getenv("MQTT_USERNAME", "Yash")
        self.mqtt_password = os.getenv("MQTT_PASSWORD", "akshitasabat")
        self.thermal_topic = "mine-rover/thermal"

        # Authoritative Live Thermal State
        self.latest_thermal_frame = np.zeros((24, 32), dtype=np.float32)
        self.latest_thermal_min = 0.0
        self.latest_thermal_max = 0.0
        self.thermal_frame_received = False
        self.thermal_last_update = 0.0

        # Threading Locks & Decoupled State
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.latest_thermal_detections = {
            "source": "MLX90640",
            "resolution": "32x24",
            "yolo_status": "WAITING FOR THERMAL DATA",
            "persons": 0,
            "detections": []
        }

        # YOLOv6 Detector Config (Secondary Consumer)
        self.weights_path = os.getenv("YOLOV6_WEIGHTS", os.path.join(YOLOV6_DIR, "yolov6s.pt"))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.detector = None
        self.stride = 32
        self.img_size = 512
        self.conf_thres = 0.20
        self.iou_thres = 0.45

        # Matplotlib Persistent Setup
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 7))

        self.image = self.ax.imshow(
            self.latest_thermal_frame,
            interpolation="bicubic",
            origin="upper",
            aspect="auto"
        )

        self.colorbar = self.fig.colorbar(
            self.image,
            ax=self.ax
        )
        self.colorbar.set_label("Temperature (°C)")

        self.ax.set_title("BhoomiGat Yaan — Live Thermal View")
        self.ax.set_xlabel("Thermal X")
        self.ax.set_ylabel("Thermal Y")

        # Initial render of placeholder state into self.latest_jpeg
        self._render_matplotlib_frame()

        # Matplotlib Animation reference
        self.anim = FuncAnimation(
            self.fig,
            self.update,
            interval=50,
            blit=False,
            cache_frame_data=False
        )

        self.mqtt_client = None
        self.running = False

        self._init_detector()

    def _render_matplotlib_frame(self):
        """Updates persistent Matplotlib figure with latest_thermal_frame and renders JPEG bytes."""
        self.image.set_data(self.latest_thermal_frame)

        low = self.latest_thermal_min
        high = self.latest_thermal_max

        if high - low < 2:
            low -= 1
            high += 1

        self.image.set_clim(low, high)

        self.ax.set_title(
            "BhoomiGat Yaan — "
            f"Thermal {low:.1f}°C – {high:.1f}°C"
        )

        buf = io.BytesIO()
        self.fig.savefig(buf, format='jpeg', bbox_inches='tight')
        self.latest_jpeg = buf.getvalue()

    def update(self, frame_idx=None):
        """Periodic Matplotlib update method."""
        with self.lock:
            self._render_matplotlib_frame()

    def _init_detector(self):
        if not YOLOV6_AVAILABLE:
            print("[YOLO] Warning: YOLOv6 modules unavailable.")
            return

        try:
            print(f"[YOLO] Loading model weights from: {self.weights_path}")
            self.detector = DetectBackend(self.weights_path, device=self.device)
            self.stride = self.detector.stride
            self.detector.model.float()
            self.detector.model.eval()
            print("[YOLO] Thermal YOLOv6 detector ready.")
        except Exception as e:
            print(f"[YOLO] Error loading YOLOv6 model: {e}")

    def start(self):
        if self.running:
            return
        self.running = True

        # Single persistent MQTT Client connection
        try:
            self.mqtt_client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id="BGY_Thermal_Backend"
            )
            self.mqtt_client.username_pw_set(self.mqtt_username, self.mqtt_password)

            ssl_ctx = ssl.create_default_context()
            self.mqtt_client.tls_set_context(ssl_ctx)

            self.mqtt_client.on_connect = self._on_connect
            self.mqtt_client.on_message = self._on_message

            print(f"[MQTT] Connecting to HiveMQ Broker: {self.mqtt_host}:{self.mqtt_port}")
            self.mqtt_client.connect_async(self.mqtt_host, self.mqtt_port, keepalive=60)
            self.mqtt_client.loop_start()
        except Exception as e:
            print(f"[MQTT] Error initializing client: {e}")

    def stop(self):
        self.running = False
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        print("[THERMAL MQTT]\nConnected")
        print(f"[THERMAL MQTT]\nSubscribed: {self.thermal_topic}")
        client.subscribe(self.thermal_topic)

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(
                msg.payload.decode()
            )

            frame = np.array(
                data["frame"],
                dtype=np.float32
            )

            if frame.size != 768:
                return

            thermal_data = frame.reshape(
                24,
                32
            )

            latest_min = float(
                np.min(thermal_data)
            )

            latest_max = float(
                np.max(thermal_data)
            )

            print(f"[THERMAL MQTT]\nFrame received:\nshape={thermal_data.shape}\nmin={latest_min:.1f}\nmax={latest_max:.1f}")

            with self.lock:
                self.latest_thermal_frame = thermal_data
                self.latest_thermal_min = latest_min
                self.latest_thermal_max = latest_max
                self.thermal_frame_received = True
                self.thermal_last_update = time.time()
                self._render_matplotlib_frame()

            # Secondary consumer: optional thermal YOLO detection
            threading.Thread(target=self._run_secondary_yolo, args=(thermal_data,), daemon=True).start()

        except Exception as e:
            print(f"[THERMAL MQTT] Message parse error: {e}")

    def _run_secondary_yolo(self, thermal_data):
        """Secondary YOLO consumer: runs YOLOv6 on thermal data without modifying Matplotlib thermal reconstruction."""
        if self.detector is None:
            return

        try:
            min_val = np.min(thermal_data)
            max_val = np.max(thermal_data)
            norm = (thermal_data - min_val) / (max_val - min_val + 1e-5)
            uint8_img = (norm * 255.0).astype(np.uint8)
            resized = cv2.resize(uint8_img, (640, 480), interpolation=cv2.INTER_CUBIC)
            yolo_image = cv2.applyColorMap(resized, cv2.COLORMAP_INFERNO)

            img = letterbox(yolo_image, self.img_size, stride=self.stride)[0]
            img = img.transpose((2, 0, 1))[::-1]
            img_tensor = torch.from_numpy(np.ascontiguousarray(img)).float() / 255.0

            if len(img_tensor.shape) == 3:
                img_tensor = img_tensor[None]

            img_tensor = img_tensor.to(self.device)

            with torch.no_grad():
                pred_results = self.detector(img_tensor)
                det = non_max_suppression(
                    pred_results,
                    conf_thres=self.conf_thres,
                    iou_thres=self.iou_thres,
                    classes=[0],  # COCO Class 0 = Person
                    agnostic=False,
                    max_det=100
                )[0]

            detections = []
            if len(det):
                det[:, :4] = Inferer.rescale(img_tensor.shape[2:], det[:, :4], yolo_image.shape).round()
                for *xyxy, conf, cls in det:
                    if int(cls) == 0:
                        x1, y1, x2, y2 = [int(v.item()) for v in xyxy]
                        detections.append({
                            "class": "person",
                            "confidence": round(float(conf.item()), 2),
                            "bbox": [x1, y1, x2, y2]
                        })

            status_text = "INFERENCE RUNNING" if len(detections) > 0 else "INFERENCE RUNNING (0 Detections)"

            with self.lock:
                self.latest_thermal_detections = {
                    "source": "MLX90640",
                    "resolution": "32x24",
                    "yolo_status": status_text,
                    "device": self.device_str,
                    "persons": len(detections),
                    "detections": detections
                }
        except Exception as e:
            print(f"[Thermal YOLO] Exception: {e}")

    def get_latest_jpeg(self):
        with self.lock:
            if self.thermal_frame_received:
                print(f"[THERMAL VIDEO]\nUsing frame:\nshape={self.latest_thermal_frame.shape}\nmin={self.latest_thermal_min:.1f}\nmax={self.latest_thermal_max:.1f}")
            return self.latest_jpeg

    def get_latest_detection(self):
        with self.lock:
            return self.latest_thermal_detections


# Singleton instance
thermal_service = MLX90640ThermalService()
