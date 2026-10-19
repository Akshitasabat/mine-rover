import { API_BASE_URL } from './modelService';

export const VIDEO_STREAM_URL = `${API_BASE_URL}/video`;
export const DETECTIONS_URL = `${API_BASE_URL}/detections`;

export const THERMAL_VIDEO_STREAM_URL = `${API_BASE_URL}/thermal/video`;
export const THERMAL_DETECTIONS_URL = `${API_BASE_URL}/thermal/detections`;

/**
 * Fetches latest RGB YOLOv6 detection data from FastAPI backend
 */
export async function fetchDetections() {
  try {
    const res = await fetch(DETECTIONS_URL);
    if (!res.ok) throw new Error('Detections API request failed');
    return await res.json();
  } catch (err) {
    console.warn('[Video Service] Error fetching live detections:', err);
    return { count: 0, persons: [] };
  }
}

/**
 * Fetches latest MLX90640 thermal YOLOv6 detection data from FastAPI backend
 */
export async function fetchThermalDetections() {
  try {
    const res = await fetch(THERMAL_DETECTIONS_URL);
    if (!res.ok) throw new Error('Thermal Detections API request failed');
    return await res.json();
  } catch (err) {
    console.warn('[Video Service] Error fetching thermal detections:', err);
    return {
      source: 'MLX90640',
      resolution: '32x24',
      persons: 0,
      detections: []
    };
  }
}
