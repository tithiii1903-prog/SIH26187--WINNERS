import React, { useState, useRef, useEffect } from 'react';
import { getFaceCameraStreamUrl, pushFaceCameraFrame } from '../services/api';
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
  const [useBrowserCam, setUseBrowserCam] = useState<boolean>(true);
  const [camPermissionError, setCamPermissionError] = useState<boolean>(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  const isRunning = status?.is_running || false;
  const streamUrl = isRunning ? getFaceCameraStreamUrl() : null;

  const requestCameraAccess = async () => {
    try {
      setCamPermissionError(false);
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
      }
      return stream;
    } catch (err) {
      console.warn("Browser camera access notice:", err);
      setCamPermissionError(true);
      return null;
    }
  };

  useEffect(() => {
    let activeStream: MediaStream | null = null;
    let pushInterval: any = null;

    if (isRunning) {
      requestCameraAccess().then((stream) => {
        activeStream = stream;
        if (stream) {
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');

          pushInterval = setInterval(() => {
            if (videoRef.current && ctx) {
              const video = videoRef.current;
              if (video.videoWidth > 0 && video.videoHeight > 0) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                canvas.toBlob((blob) => {
                  if (blob) {
                    pushFaceCameraFrame(blob);
                  }
                }, 'image/jpeg', 0.7);
              }
            }
          }, 120);
        }
      });
    }

    return () => {
      if (pushInterval) clearInterval(pushInterval);
      if (activeStream) {
        activeStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [isRunning]);

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
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <button
                className={`cctv-btn compact ${useBrowserCam ? 'primary' : ''}`}
                onClick={() => setUseBrowserCam(!useBrowserCam)}
                title="Toggle client browser webcam"
                style={{ fontSize: '0.75rem', padding: '0.2rem 0.5rem' }}
              >
                {useBrowserCam ? '📷 WebCam: ON' : '📷 WebCam'}
              </button>
              <button
                className="cctv-btn danger compact"
                onClick={handleStop}
                disabled={loading}
                id="hd-camera-stop-btn"
              >
                <span className="btn-icon">■</span>
                <span>{loading ? 'Stopping...' : 'Stop HD'}</span>
              </button>
            </div>
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
        {isRunning ? (
          <>
            <video
              ref={videoRef}
              className="hd-stream-img"
              autoPlay
              playsInline
              muted
              style={{
                display: useBrowserCam ? 'block' : 'none',
                width: '100%',
                height: '100%',
                objectFit: 'contain'
              }}
            />
            {streamUrl && (
              <img
                src={streamUrl}
                alt="HD Face Recognition Live Stream"
                className="hd-stream-img"
                style={{
                  display: !useBrowserCam ? 'block' : 'none',
                  width: '100%',
                  height: '100%',
                  objectFit: 'contain'
                }}
              />
            )}
            {camPermissionError && (
              <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', zIndex: 10 }}>
                <button className="cctv-btn primary compact" onClick={requestCameraAccess}>
                  📷 Enable Device Camera Access
                </button>
              </div>
            )}
          </>
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
