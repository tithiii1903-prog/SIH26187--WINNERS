import React from 'react';
import type { Feed, LiveAnalytics } from '../services/api';

interface SystemStatusProps {
  feed: Feed | null;
  analytics: LiveAnalytics | null;
  isLive: boolean;
  onStart: () => void;
  onStop: () => void;
}

const SystemStatus: React.FC<SystemStatusProps> = ({ feed, analytics, isLive, onStart, onStop }) => {
  if (!feed) {
    return (
      <div className="panel">
        <div className="panel-header">Controls</div>
        <div className="empty-state" style={{ padding: '1rem', minHeight: 'unset' }}>
          Select a feed to view controls
        </div>
      </div>
    );
  }


  return (
    <div className="panel">
      <div className="panel-header">Controls & Status</div>
      <table className="status-table">
        <tbody>
          <tr>
            <td>Feed Status</td>
            <td>
              <span className={`status-indicator ${isLive ? 'success' : (feed.status === 'ERROR' ? 'danger' : 'neutral')}`}>
                <span className="status-dot" />
                {feed.status}
              </span>
            </td>
          </tr>
          <tr>
            <td>Processing (AI) FPS</td>
            <td style={{ color: 'var(--accent-color)' }}>
              {analytics && isLive ? analytics.processing_fps.toFixed(1) : '—'}
            </td>
          </tr>
          <tr>
            <td>Stream FPS</td>
            <td style={{ color: '#00ffcc' }}>
              {analytics && isLive && analytics.stream_fps ? analytics.stream_fps.toFixed(1) : (isLive ? '30.0' : '—')}
            </td>
          </tr>
          <tr>
            <td>Source FPS</td>
            <td>{feed.fps}</td>
          </tr>
          <tr>
            <td>Frames Processed</td>
            <td>{analytics && isLive ? analytics.frames_processed : '—'}</td>
          </tr>
          <tr>
            <td>Resolution</td>
            <td>{feed.width}×{feed.height}</td>
          </tr>
        </tbody>
      </table>

      <div style={{ marginTop: '0.75rem', display: 'flex', gap: '0.5rem' }}>
        <button
          className="btn primary"
          style={{ flex: 1, justifyContent: 'center' }}
          onClick={onStart}
          disabled={isLive || feed.status === 'STARTING'}
        >
          {feed.status === 'STARTING' ? '⏳ Starting...' : '▶ START'}
        </button>
        <button
          className="btn danger"
          style={{ flex: 1, justifyContent: 'center' }}
          onClick={onStop}
          disabled={!isLive && feed.status !== 'STARTING'}
        >
          ■ STOP
        </button>
      </div>
    </div>
  );
};

export default SystemStatus;
