import React, { useState, useEffect } from 'react';
import { Eye, UserCheck, Crosshair, Camera, Flame, ShieldAlert, Cpu } from 'lucide-react';
import {
  VIDEO_STREAM_URL,
  fetchDetections,
  THERMAL_VIDEO_STREAM_URL,
  fetchThermalDetections
} from '../services/videoService';

export default function AiVisionWorkstation() {
  const [visionMode, setVisionMode] = useState('live_camera'); // 'live_camera' | 'thermal' | 'combined'
  const [showBoundingBox, setShowBoundingBox] = useState(true);

  // Live RGB YOLOv6 Detection State from FastAPI /detections (CP PLUS)
  const [detections, setDetections] = useState({
    count: 0,
    persons: []
  });

  // Live Thermal YOLOv6 Detection State from FastAPI /thermal/detections (MLX90640)
  const [thermalDetections, setThermalDetections] = useState({
    source: "MLX90640",
    resolution: "32x24",
    persons: 0,
    detections: []
  });

  // Poll live RGB & Thermal detections from FastAPI backend every 400ms
  useEffect(() => {
    let isMounted = true;
    const updateAllDetections = async () => {
      const rgbData = await fetchDetections();
      const thermalData = await fetchThermalDetections();
      if (isMounted) {
        setDetections(rgbData || { count: 0, persons: [] });
        setThermalDetections(thermalData || { source: "MLX90640", resolution: "32x24", persons: 0, detections: [] });
      }
    };

    updateAllDetections();
    const interval = setInterval(updateAllDetections, 400);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const thermalPersonCount = thermalDetections.persons || 0;
  const isThermalLock = thermalPersonCount > 0;

  return (
    <section id="ai-vision" className="py-24 bg-parchment bg-newspaper-dots text-walnut-dark relative border-b-2 border-walnut">
      <div className="max-w-[1200px] mx-auto px-6 md:px-10">
        
        {/* Header */}
        <div className="border-b-2 border-walnut pb-6 mb-12 flex flex-wrap items-end justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-bronze uppercase tracking-widest mb-2">
              <Eye className="w-4 h-4 text-walnut" />
              <span>SECTION 06 — DUAL VISION: CP-E23A & MLX90640 THERMAL CAMERA</span>
            </div>
            <h2 className="font-serif text-3xl md:text-4xl font-bold text-walnut-dark tracking-tight">
              SUBTERRANEAN WORKER RECOGNITION
            </h2>
          </div>
          <p className="font-sans text-sm text-walnut/80 max-w-md">
            Parallel stream processing of CP PLUS optical RTSP and MLX90640 MQTT thermal matrix with PyTorch YOLOv6 edge detection.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* LEFT: Vision Mode Controls & Human Detection Analytics */}
          <div className="lg:col-span-5 space-y-6">
            
            {/* Mode Selector Card */}
            <div className="bg-parchment-paper border border-walnut p-6">
              <span className="font-mono text-xs font-bold text-walnut/60 block mb-3 uppercase">
                CAMERA STREAM & THERMAL SELECTOR
              </span>

              <div className="space-y-2 mb-6">
                {[
                  { id: 'live_camera', label: 'CP-E23A LIVE RTSP CAMERA (YOLOv6)', desc: 'Real-time OpenCV Stream + YOLOv6-S Bounding' },
                  { id: 'thermal', label: 'MLX90640 LIVE MQTT THERMAL (YOLOv6)', desc: 'Reconstructed 32x24 Heatmap via HiveMQ MQTT' },
                  { id: 'combined', label: 'AI FUSED OPTICAL & THERMAL', desc: 'Dual-Layer Stream with Heat Signature Overlay' },
                ].map((mode) => (
                  <button
                    key={mode.id}
                    onClick={() => setVisionMode(mode.id)}
                    className={`w-full text-left p-3 font-mono text-xs border transition-colors flex flex-col ${
                      visionMode === mode.id
                        ? 'bg-walnut text-parchment border-walnut font-bold'
                        : 'bg-parchment border-walnut/40 text-walnut hover:border-walnut'
                    }`}
                  >
                    <span className="text-amber font-bold">{mode.label}</span>
                    <span className="text-[10px] text-parchment/70 font-normal mt-0.5">{mode.desc}</span>
                  </button>
                ))}
              </div>

              {/* Toggles */}
              <div className="space-y-2 pt-4 border-t border-walnut/30 font-mono text-xs">
                <label className="flex items-center justify-between cursor-pointer">
                  <span>SHOW AI BOUNDING BOXES</span>
                  <input
                    type="checkbox"
                    checked={showBoundingBox}
                    onChange={(e) => setShowBoundingBox(e.target.checked)}
                    className="accent-walnut w-4 h-4"
                  />
                </label>
              </div>
            </div>

            {/* AI Human Detection Summary Card */}
            <div className="bg-walnut text-parchment border-2 border-walnut p-6">
              <div className="flex items-center justify-between border-b border-bronze pb-3 mb-4">
                <div className="flex items-center gap-2">
                  <UserCheck className="w-4 h-4 text-amber" />
                  <span className="font-mono text-xs font-bold text-amber">HUMAN DETECTION STATUS</span>
                </div>
                <span className="font-mono text-[11px] bg-amber text-walnut font-bold px-2 py-0.5">
                  Persons: {detections.count}
                </span>
              </div>

              {/* RGB CP PLUS Detection Details */}
              <div className="space-y-3 font-mono text-xs mb-4">
                <div className="text-[11px] text-amber font-bold uppercase tracking-wider border-b border-walnut-dark pb-1">
                  CP PLUS Optical Camera
                </div>
                {detections.count > 0 ? (
                  detections.persons.map((person, idx) => (
                    <div key={idx} className="bg-walnut-dark p-2.5 border border-amber/40 rounded-sm">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-amber font-bold">Worker #{idx + 1}</span>
                        <span className="text-emerald-400 font-bold">
                          Confidence: {(person.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="flex justify-between text-[10px] text-parchment/60">
                        <span>BBox:</span>
                        <span className="text-parchment font-mono">
                          [{person.bbox ? person.bbox.join(', ') : 'N/A'}]
                        </span>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="py-2 text-center text-[11px] text-parchment/60 border border-dashed border-bronze/40">
                    No RGB optical detection.
                  </div>
                )}
              </div>

              {/* Thermal MLX90640 Detection Details */}
              <div className="space-y-3 font-mono text-xs mb-4 pt-2 border-t border-bronze/40">
                <div className="flex justify-between items-center text-[11px] text-amber font-bold uppercase tracking-wider border-b border-walnut-dark pb-1">
                  <span>MLX90640 THERMAL FEED — MQTT HIVEMQ</span>
                  <span className={isThermalLock ? "text-emerald-400 font-bold" : "text-parchment/60"}>
                    Thermal Lock: {isThermalLock ? "YES" : "NO"}
                  </span>
                </div>
                <div className="flex justify-between text-[11px] font-mono">
                  <span className="text-parchment/60">Persons:</span>
                  <span className="text-amber font-bold">{thermalPersonCount}</span>
                </div>

                {isThermalLock ? (
                  thermalDetections.detections.map((det, idx) => (
                    <div key={idx} className="bg-walnut-dark p-2.5 border border-emerald-500/40 rounded-sm">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-emerald-400 font-bold">PERSON #{idx + 1}</span>
                        <span className="text-emerald-400 font-bold">
                          Confidence: {(det.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="flex justify-between text-[10px] text-parchment/60">
                        <span>BBox:</span>
                        <span className="text-parchment font-mono">
                          [{det.bbox ? det.bbox.join(', ') : 'N/A'}]
                        </span>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="py-2 text-center text-[11px] text-parchment/60 border border-dashed border-bronze/40">
                    No thermal person detected.
                  </div>
                )}
              </div>

              <div className="space-y-2 font-mono text-[11px] mb-4 text-parchment/80 border-t border-bronze/40 pt-3">
                <div className="flex justify-between">
                  <span className="text-parchment/60">RGB RTSP SOURCE:</span>
                  <span className="text-emerald-400">192.168.0.104:5543</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-parchment/60">THERMAL MQTT BROKER:</span>
                  <span className="text-amber">HiveMQ Cloud (mine-rover/thermal)</span>
                </div>
              </div>

              <p className="font-sans text-xs text-parchment/80 bg-walnut-dark p-3 border border-bronze leading-normal">
                MLX90640 32x24 thermal matrix reconstructed live with Matplotlib and fed into YOLOv6 detector.
              </p>
            </div>

          </div>

          {/* RIGHT: Live Camera & Thermal Viewport */}
          <div className="lg:col-span-7 bg-walnut-dark border-2 border-walnut p-6 text-parchment relative min-h-[480px] shadow-2xl flex flex-col justify-between">
            
            {/* Viewport Overlay Header */}
            <div className="flex items-center justify-between border-b border-bronze pb-3 font-mono text-xs text-amber">
              <div className="flex items-center gap-2">
                {visionMode === 'thermal' ? (
                  <Flame className="w-4 h-4 text-amber animate-pulse" />
                ) : (
                  <Camera className="w-4 h-4 text-amber animate-pulse" />
                )}
                <span>
                  {visionMode === 'thermal'
                    ? 'MLX90640 THERMAL FEED — MQTT HIVEMQ'
                    : 'CP-E23A LIVE FEED — 192.168.0.104:5543'}
                </span>
              </div>
              <span className="text-emerald-400">
                {visionMode === 'thermal'
                  ? (thermalDetections.yolo_status || 'YOLOv6: INFERENCE RUNNING')
                  : 'YOLOv6-S ONLINE'}
              </span>
            </div>

            {/* Viewport Screen */}
            <div className="relative my-6 aspect-video bg-black border border-bronze overflow-hidden flex items-center justify-center">
              
              {/* LIVE CP-E23A ANNOTATED CAMERA STREAM FROM FASTAPI */}
              {visionMode === 'live_camera' && (
                <img
                  src={VIDEO_STREAM_URL}
                  alt="CP-E23A Live YOLOv6 Stream"
                  className="w-full h-full object-contain bg-black"
                  onError={(e) => {
                    e.target.onerror = null;
                    e.target.src = '';
                  }}
                />
              )}

              {/* LIVE MLX90640 MATPLOTLIB RECONSTRUCTED THERMAL STREAM FROM FASTAPI */}
              {visionMode === 'thermal' && (
                <img
                  src={THERMAL_VIDEO_STREAM_URL}
                  alt="MLX90640 Live Matplotlib Thermal Stream"
                  className="w-full h-full object-contain bg-black"
                  onError={(e) => {
                    e.target.onerror = null;
                    e.target.src = '';
                  }}
                />
              )}

              {/* Combined Mode View */}
              {visionMode === 'combined' && (
                <div className="relative w-full h-full flex">
                  <div className="w-1/2 h-full border-r border-bronze">
                    <img
                      src={VIDEO_STREAM_URL}
                      alt="CP-E23A Live Stream"
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="w-1/2 h-full">
                    <img
                      src={THERMAL_VIDEO_STREAM_URL}
                      alt="Thermal Stream"
                      className="w-full h-full object-cover"
                    />
                  </div>
                </div>
              )}

              {/* Crosshair Overlay */}
              <div className="absolute inset-0 pointer-events-none flex items-center justify-center opacity-20">
                <div className="w-full h-[1px] bg-amber" />
                <div className="h-full w-[1px] bg-amber absolute" />
              </div>
            </div>

            {/* Viewport Footer Data Bar */}
            <div className="grid grid-cols-3 gap-2 pt-3 border-t border-bronze font-mono text-[10px] text-parchment/70">
              <div>
                {visionMode === 'thermal' ? 'SENSOR: MLX90640 (32x24)' : 'RTSP PATH: /live/channel0'}
              </div>
              <div className="text-center text-amber">
                {visionMode === 'thermal'
                  ? `Thermal Lock: ${isThermalLock ? 'YES' : 'NO'} | Persons: ${thermalPersonCount}`
                  : `PERSONS DETECTED: ${detections.count}`}
              </div>
              <div className="text-right">YOLO WEIGHTS: yolov6s.pt</div>
            </div>

          </div>

        </div>

      </div>
    </section>
  );
}
