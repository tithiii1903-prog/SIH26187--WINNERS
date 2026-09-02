import React from 'react';

interface PersonAnalyticsProps {
  currentPersons: number | string;
  peakPersons: number | string;
  activeTracks: number | string;
}

const PersonAnalytics: React.FC<PersonAnalyticsProps> = ({
  currentPersons,
  peakPersons,
  activeTracks,
}) => {
  return (
    <div className="panel command-panel analytics-card" id="person-analytics-panel">
      <div className="panel-header">
        <span className="panel-title">👤 PERSON SURVEILLANCE TELEMETRY</span>
      </div>

      <div className="analytics-metrics-grid">
        <div className="telemetry-metric-tile">
          <span className="metric-tag">CURRENT DETECTIONS</span>
          <span className="metric-num highlight">{currentPersons}</span>
        </div>
        <div className="telemetry-metric-tile">
          <span className="metric-tag">PEAK CONCURRENT</span>
          <span className="metric-num cyan">{peakPersons}</span>
        </div>
        <div className="telemetry-metric-tile">
          <span className="metric-tag">BYTETRACK TARGETS</span>
          <span className="metric-num warning">{activeTracks}</span>
        </div>
      </div>
    </div>
  );
};

export default PersonAnalytics;
