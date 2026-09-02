import React from 'react';
import type { FaceDetection } from '../services/api';

interface WatchlistAlertProps {
  faceMatches: FaceDetection[];
}

const WatchlistAlert: React.FC<WatchlistAlertProps> = ({ faceMatches }) => {
  const confirmedMatches = faceMatches.filter(f => f.matched && f.name);
  const hasMatches = confirmedMatches.length > 0;
  const hasCritical = confirmedMatches.some(f => f.status === 'CRITICAL');

  return (
    <div
      className={`panel command-panel watchlist-alert-deck ${
        hasCritical ? 'critical-active' : hasMatches ? 'warning-active' : ''
      }`}
      id="watchlist-alert-panel"
    >
      <div className="panel-header">
        <span className="panel-title">
          {hasCritical ? '🚨 CRITICAL WATCHLIST ALARM' : hasMatches ? '⚠️ WATCHLIST MATCH DETECTED' : 'FACIAL WATCHLIST MONITOR'}
        </span>
        <span className={`status-indicator ${hasCritical ? 'danger' : hasMatches ? 'warning' : 'neutral'}`}>
          <span className="status-dot" />
          {hasCritical ? 'CRITICAL' : hasMatches ? 'MATCH' : 'CLEAR'}
        </span>
      </div>

      {hasMatches ? (
        <div className="scrollable-list match-list-deck">
          {confirmedMatches.map((m) => {
            const isCrit = m.status === 'CRITICAL';
            return (
              <div
                key={`${m.face_id}-${m.name}`}
                className={`watchlist-match-card ${isCrit ? 'critical' : 'warning'}`}
              >
                <div className="match-card-header">
                  <span className={`match-badge ${isCrit ? 'crit' : 'warn'}`}>
                    {isCrit ? '🚨 CRITICAL MATCH' : '🔴 WATCHLIST MATCH'}
                  </span>
                  <span className="match-face-id">FACE #{m.face_id}</span>
                </div>

                <div className="match-identity-row">
                  <span className="match-person-name">{m.name}</span>
                  <span className="match-similarity">
                    {(m.similarity * 100).toFixed(1)}% SIM
                  </span>
                </div>

                <div className="match-footer-meta">
                  <span className="match-status-tag">{m.status}</span>
                  <span className="match-cam-tag">HD FACE SUBSYSTEM</span>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="empty-state-banner">
          <span className="empty-icon">🛡️</span>
          <span>No active watchlist alerts detected on sensor streams.</span>
        </div>
      )}
    </div>
  );
};

export default WatchlistAlert;
