import React from 'react';

interface IntrusionPanelProps {
  activeIntrusions: number[];
}

const IntrusionPanel: React.FC<IntrusionPanelProps> = ({ activeIntrusions }) => {
  const hasIntrusions = activeIntrusions.length > 0;

  return (
    <div
      className={`panel command-panel intrusion-deck ${hasIntrusions ? 'breach-active' : ''}`}
      id="intrusion-monitor-panel"
    >
      <div className="panel-header">
        <span className="panel-title">VIRTUAL FENCE INTRUSION RADAR</span>
        <span className={`status-indicator ${hasIntrusions ? 'danger' : 'neutral'}`}>
          <span className="status-dot" />
          {hasIntrusions ? 'BREACH DETECTED' : 'SECURE'}
        </span>
      </div>

      {hasIntrusions ? (
        <div className="scrollable-list intrusion-list">
          {activeIntrusions.map(trackId => (
            <div key={trackId} className="intrusion-alert-card">
              <div className="intrusion-header">
                <span className="intrusion-tag">🚨 PERIMETER BREACH</span>
                <span className="intrusion-track-id">TRACK ID #{trackId}</span>
              </div>
              <div className="intrusion-body">
                <span>Subject entered Restricted Border Zone polygon</span>
                <span className="intrusion-live-pulse">ACTIVE TARGET</span>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="empty-state-banner">
          <span className="empty-icon">🛡️</span>
          <span>Restricted border perimeter secure. Zero active fence intrusions.</span>
        </div>
      )}
    </div>
  );
};

export default IntrusionPanel;
