import React from 'react';

interface VehicleAnalyticsProps {
  currentVehicles: number | string;
  peakVehicles: number | string;
  cars: number | string;
  motorcycles: number | string;
  buses: number | string;
  trucks: number | string;
}

const VehicleAnalytics: React.FC<VehicleAnalyticsProps> = ({
  currentVehicles,
  peakVehicles,
  cars,
  motorcycles,
  buses,
  trucks,
}) => {
  return (
    <div className="panel command-panel analytics-card" id="vehicle-analytics-panel">
      <div className="panel-header">
        <span className="panel-title">🚗 VEHICLE CLASSIFICATION TELEMETRY</span>
      </div>

      <div className="analytics-metrics-grid vehicles-grid">
        <div className="telemetry-metric-tile">
          <span className="metric-tag">CURRENT VEHICLES</span>
          <span className="metric-num cyan">{currentVehicles}</span>
        </div>
        <div className="telemetry-metric-tile">
          <span className="metric-tag">PEAK VEHICLES</span>
          <span className="metric-num">{peakVehicles}</span>
        </div>
        <div className="telemetry-metric-tile">
          <span className="metric-tag">CARS</span>
          <span className="metric-num highlight">{cars}</span>
        </div>
        <div className="telemetry-metric-tile">
          <span className="metric-tag">MOTORCYCLES</span>
          <span className="metric-num">{motorcycles}</span>
        </div>
        <div className="telemetry-metric-tile">
          <span className="metric-tag">BUSES</span>
          <span className="metric-num">{buses}</span>
        </div>
        <div className="telemetry-metric-tile">
          <span className="metric-tag">TRUCKS</span>
          <span className="metric-num">{trucks}</span>
        </div>
      </div>
    </div>
  );
};

export default VehicleAnalytics;
