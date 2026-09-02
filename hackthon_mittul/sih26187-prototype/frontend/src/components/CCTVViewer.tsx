import React, { useRef } from 'react';
import { getMjpegStreamUrl } from '../services/api';
import type { Feed, Zone, LiveAnalytics } from '../services/api';
import FenceEditor from './FenceEditor';

export type CCTVPlayerStatus = 'IDLE' | 'READY' | 'STARTING' | 'LIVE' | 'STOPPING' | 'STOPPED' | 'ERROR';

interface CCTVViewerProps {
  feed: Feed | null;
  status: CCTVPlayerStatus;
  isLive: boolean;
  streamSessionId: string | number | null;
  onStart: () => void;
  onStop: () => void;
  onReplay: () => void;
  initialZone: Zone | null;
  onZoneSaved: () => void;
  isFenceEnabled: boolean;
  onToggleFenceEnable: (enabled: boolean) => void;
  hasIntrusions?: boolean;
  analytics: LiveAnalytics | null;
  errorMessage?: string | null;
}

const CCTVViewer: React.FC<CCTVViewerProps> = ({
  feed,
  status,
  isLive,
  streamSessionId,
  onStart,
  onStop,
  onReplay,
  initialZone,
  onZoneSaved,
  isFenceEnabled,
  onToggleFenceEnable,
  hasIntrusions = false,
  analytics,
  errorMessage,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  // Compute stream URL only when explicitly LIVE with valid session ID
  const streamUrl = feed && isLive && streamSessionId !== null
    ? getMjpegStreamUrl(feed.id, streamSessionId)
    : null;

  const isBusy = status === 'STARTING' || status === 'STOPPING';
  const isMp4 = (feed?.source_type || 'file') === 'file';

  return (
    <div className="cctv-command-viewport-card">
      {/* Viewport Header */}
      <div className="cctv-viewport-header">
        <div className="cctv-header-left">
          <span className="cctv-title-tag">PRIMARY SURVEILLANCE FEED</span>
          {feed && (
            <span className="cctv-feed-name-tag">
              {feed.name}
            </span>
          )}
        </div>

        <div className="cctv-header-right">
          {feed && (
            <span className="cctv-source-type-badge">
              {feed.source_type === 'camera'
                ? `📷 DEVICE ${feed.device_index ?? 0}`
                : `🎥 MP4 SOURCE (${feed.width}×${feed.height})`}
            </span>
          )}

          {/* Status Indicator */}
          <div className={`cctv-status-pill ${status.toLowerCase()}`}>
            <span className="status-dot" />
            <span className="status-text">
              {status === 'LIVE' ? 'LIVE SURVEILLANCE' : status}
            </span>
          </div>
        </div>
      </div>

      {/* Main Video Display Canvas Container */}
      <div ref={containerRef} className="cctv-viewport-canvas" id="cctv-video-viewport">
        {/* State A: IDLE (No Feed Selected) */}
        {status === 'IDLE' && (
          <div className="cctv-state-view idle-state">
            <div className="cctv-state-icon">📡</div>
            <div className="cctv-state-title">NO SURVEILLANCE FEED SELECTED</div>
            <div className="cctv-state-desc">
              Select a recorded MP4 feed or connect a device camera from the Sources panel to begin real-time analytics.
            </div>
          </div>
        )}

        {/* State B: READY (Feed selected, ready to start) */}
        {status === 'READY' && feed && (
          <div className="cctv-state-view ready-state">
            <div className="cctv-state-icon">📹</div>
            <div className="cctv-state-title">FEED READY: {feed.name}</div>
            <div className="cctv-state-specs">
              <span>{feed.width}×{feed.height} Resolution</span>
              <span>•</span>
              <span>{feed.fps} FPS Source</span>
              {feed.duration && (
                <>
                  <span>•</span>
                  <span>{feed.duration.toFixed(1)}s Duration</span>
                </>
              )}
            </div>
            <button
              className="cctv-start-hero-btn"
              onClick={onStart}
              disabled={isBusy}
              id="cctv-hero-start-btn"
            >
              <span className="btn-icon">▶</span> START SURVEILLANCE
            </button>
          </div>
        )}

        {/* State C: STARTING (Initializing backend worker) */}
        {status === 'STARTING' && (
          <div className="cctv-state-view starting-state">
            <div className="cctv-loading-ring" />
            <div className="cctv-state-title">INITIALIZING AI SURVEILLANCE PIPELINE</div>
            <div className="cctv-state-desc">
              Allocating YOLOv8n detector, ByteTracker, and stream pipeline...
            </div>
          </div>
        )}

        {/* State D: LIVE (MJPEG stream mounted) */}
        {status === 'LIVE' && streamUrl && (
          <>
            <img
              key={`stream-${feed?.id}-${streamSessionId}`}
              src={streamUrl}
              alt="Live CCTV AI Stream"
              className="cctv-stream"
            />
            {/* Real-time Telemetry Floating HUD */}
            <div className="cctv-live-telemetry-hud">
              <div className="hud-metric">
                <span className="hud-label">AI FPS</span>
                <span className="hud-val highlight">
                  {analytics && analytics.processing_fps ? analytics.processing_fps.toFixed(1) : '--'}
                </span>
              </div>
              <div className="hud-metric">
                <span className="hud-label">STREAM FPS</span>
                <span className="hud-val cyan">
                  {analytics && analytics.stream_fps ? analytics.stream_fps.toFixed(1) : '30.0'}
                </span>
              </div>
              <div className="hud-metric">
                <span className="hud-label">SRC FPS</span>
                <span className="hud-val">{feed?.fps || '--'}</span>
              </div>
              <div className="hud-metric">
                <span className="hud-label">TRACKS</span>
                <span className="hud-val">{analytics ? analytics.active_tracks : '--'}</span>
              </div>
            </div>
          </>
        )}

        {/* State E: STOPPING (Terminating processor) */}
        {status === 'STOPPING' && (
          <div className="cctv-state-view stopping-state">
            <div className="cctv-loading-ring danger" />
            <div className="cctv-state-title">STOPPING FEED PROCESSING</div>
            <div className="cctv-state-desc">Releasing stream buffer and pipeline threads...</div>
          </div>
        )}

        {/* State F: STOPPED / PLAYBACK COMPLETE */}
        {status === 'STOPPED' && feed && (
          <div className="cctv-state-view stopped-state">
            <div className="cctv-state-icon">⏹</div>
            <div className="cctv-state-title">
              {isMp4 ? 'PLAYBACK COMPLETE' : 'FEED STOPPED'}
            </div>
            <div className="cctv-state-desc">
              {isMp4
                ? 'Recorded video reached end-of-file. Click Replay to restart from frame 0.'
                : 'Surveillance stream has been stopped.'}
            </div>
            <div className="cctv-stopped-actions">
              <button
                className="cctv-btn primary hero"
                onClick={onReplay}
                disabled={isBusy}
                id="cctv-hero-replay-btn"
              >
                <span className="btn-icon">↻</span> {isMp4 ? 'REPLAY FROM FRAME 0' : 'RESTART FEED'}
              </button>
            </div>
          </div>
        )}

        {/* State G: ERROR */}
        {status === 'ERROR' && (
          <div className="cctv-state-view error-state">
            <div className="cctv-state-icon error">⚠️</div>
            <div className="cctv-state-title">SURVEILLANCE FEED ERROR</div>
            <div className="cctv-state-desc">
              {errorMessage || 'Failed to establish feed stream connection with backend.'}
            </div>
            <button
              className="cctv-btn danger hero"
              onClick={onStart}
              disabled={isBusy}
              id="cctv-hero-retry-btn"
            >
              <span className="btn-icon">↻</span> RETRY FEED
            </button>
          </div>
        )}

        {/* Virtual Fence Editor Overlay */}
        {feed && (
          <FenceEditor
            containerRef={containerRef}
            feedWidth={feed.width || 1280}
            feedHeight={feed.height || 720}
            initialZone={initialZone}
            onZoneSaved={onZoneSaved}
            isFenceEnabled={isFenceEnabled}
            onToggleEnable={onToggleFenceEnable}
            hasIntrusion={hasIntrusions}
          />
        )}
      </div>

      {/* Control Bar directly beneath Viewport */}
      <div className="cctv-control-bar">
        <div className="cctv-control-left">
          {/* Start / Stop / Replay Action Buttons */}
          <button
            className={`cctv-btn success ${status === 'LIVE' ? 'active-pulse' : ''}`}
            onClick={onStart}
            disabled={status === 'LIVE' || isBusy || !feed}
            title={status === 'LIVE' ? 'Feed is currently running' : 'Start live processing'}
            id="cctv-ctrl-start-btn"
          >
            <span className="btn-icon">{status === 'STARTING' ? '⏳' : '▶'}</span>
            <span>{status === 'STARTING' ? 'Starting...' : 'Start'}</span>
          </button>

          <button
            className="cctv-btn danger"
            onClick={onStop}
            disabled={status !== 'LIVE' && status !== 'STARTING'}
            title="Stop live processing"
            id="cctv-ctrl-stop-btn"
          >
            <span className="btn-icon">{status === 'STOPPING' ? '⏳' : '■'}</span>
            <span>{status === 'STOPPING' ? 'Stopping...' : 'Stop'}</span>
          </button>

          <button
            className="cctv-btn"
            onClick={onReplay}
            disabled={isBusy || !feed}
            title="Restart feed from frame 0 with clean session"
            id="cctv-ctrl-replay-btn"
          >
            <span className="btn-icon">↻</span>
            <span>Replay</span>
          </button>
        </div>

        <div className="cctv-control-right">
          <div className="cctv-telemetry-badge">
            <span className="meta-label">HUMANS</span>
            <span className="meta-value highlight">
              {analytics && isLive ? analytics.current_persons : '--'}
            </span>
          </div>

          <div className="cctv-telemetry-badge">
            <span className="meta-label">VEHICLES</span>
            <span className="meta-value cyan">
              {analytics && isLive ? analytics.current_vehicles : '--'}
            </span>
          </div>

          <div className={`cctv-telemetry-badge ${hasIntrusions ? 'danger-alert' : ''}`}>
            <span className="meta-label">INTRUSIONS</span>
            <span className={`meta-value ${hasIntrusions ? 'danger' : ''}`}>
              {analytics && isLive ? (analytics.active_intrusions ? analytics.active_intrusions.length : 0) : '--'}
            </span>
          </div>

          <div className="cctv-telemetry-badge">
            <span className="meta-label">FRAMES</span>
            <span className="meta-value">
              {analytics && isLive ? analytics.frames_processed : '--'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CCTVViewer;
