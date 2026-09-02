import React, { useState } from 'react';
import { getFaceCameraStreamUrl } from '../services/api';
import type { FaceCameraStatus, FaceCameraResults } from '../services/api';

interface HDFaceViewerProps {
  status: FaceCameraStatus | null;
  results: FaceCameraResults | null;
  onStart: (deviceIndex: number) => Promise<void>;
  onStop: () => Promise<void>;
}

const HDFaceViewer: React.FC<HDFaceViewerProps> = ({
  status,
  results,
  onStart,
  onStop,
}) => {
  const [deviceIndex, setDeviceIndex] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const isRunning = status?.is_running || false;
  const streamUrl = isRunning ? getFaceCameraStreamUrl() : null;

  const handleStart = async () => {
    setLoading(true);
    setError(null);
    try {
      await onStart(deviceIndex);
    } catch (e: any) {
      setError(e.message || 'Failed to initialize HD face camera');
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    setLoading(true);
    setError(null);
    try {
      await onStop();
    } catch (e: any) {
      setError(e.message || 'Failed to stop HD face camera');
    } finally {
      setLoading(false);
    }
  };

  const faceCount = results?.faces?.length || 0;
  const matchedFaces = results?.faces?.filter(f => f.matched) || [];

  return (
    <div className="panel command-panel hd-subsystem-card" id="hd-face-camera-subsystem">
      {/* Subsystem Header */}
      <div className="panel-header hd-header">
        <div className="hd-title-group">
          <span className="hd-subsystem-tag">SUBSYSTEM 02</span>
          <span className="hd-main-title">🎯 HD FACE RECOGNITION CAMERA</span>
        </div>

        <div className="hd-header-controls">
          <div className={`cctv-status-pill ${isRunning ? 'live' : 'offline'}`}>
            <span className="status-dot" />
            <span className="status-text">{isRunning ? 'HD ACTIVE' : 'OFFLINE'}</span>
          </div>

          {!isRunning ? (
            <div className="hd-quick-actions">
              <label className="hd-dev-label">
                DEV:
                <input
                  type="number"
                  min="0"
                  max="10"
                  value={deviceIndex}
                  onChange={(e) => setDeviceIndex(parseInt(e.target.value) || 0)}
                  className="hd-dev-input"
                  disabled={loading}
                />
              </label>
              <button
                className="cctv-btn success compact"
                onClick={handleStart}
                disabled={loading}
                id="hd-camera-start-btn"
              >
                <span className="btn-icon">▶</span>
                <span>{loading ? 'Starting...' : 'Start HD'}</span>
              </button>
            </div>
          ) : (
            <button
              className="cctv-btn danger compact"
              onClick={handleStop}
              disabled={loading}
              id="hd-camera-stop-btn"
            >
              <span className="btn-icon">■</span>
              <span>{loading ? 'Stopping...' : 'Stop HD'}</span>
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="feed-error-toast" style={{ margin: '0.4rem 0' }}>
          ⚠️ {error}
        </div>
      )}

      {/* Telemetry Bar */}
      {isRunning && status && (
        <div className="hd-telemetry-hud">
          <div className="hd-hud-item">
            <span className="hud-k">REC FPS</span>
            <span className="hud-v highlight">{status.recognition_fps.toFixed(1)}</span>
          </div>
          <div className="hd-hud-item">
            <span className="hud-k">SRC FPS</span>
            <span className="hud-v">{status.source_fps.toFixed(1)}</span>
          </div>
          <div className="hd-hud-item">
            <span className="hud-k">LATENCY</span>
            <span className="hud-v">{status.average_recognition_latency_ms.toFixed(0)} ms</span>
          </div>
          <div className="hd-hud-item">
            <span className="hud-k">VISIBLE</span>
            <span className={`hud-v ${faceCount > 0 ? 'cyan' : ''}`}>{faceCount}</span>
          </div>
          <div className="hd-hud-item">
            <span className="hud-k">MATCHES</span>
            <span className={`hud-v ${matchedFaces.length > 0 ? 'danger' : ''}`}>
              {matchedFaces.length}
            </span>
          </div>
          <div className="hd-hud-item">
            <span className="hud-k">ENROLLED</span>
            <span className="hud-v">{status.registered_faces_count}</span>
          </div>
        </div>
      )}

      {/* Viewport Display */}
      <div className="hd-viewport-box">
        {isRunning && streamUrl ? (
          <img
            src={streamUrl}
            alt="HD Face Recognition Live Stream"
            className="hd-stream-img"
          />
        ) : (
          <div className="hd-offline-placeholder">
            <div className="offline-icon">🎯</div>
            <div className="offline-title">HD FACE CAMERA DISENGAGED</div>
            <div className="offline-desc">
              RetinaFace detection & ArcFace 512D recognition are offline. Select device index and click Start HD.
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default HDFaceViewer;
