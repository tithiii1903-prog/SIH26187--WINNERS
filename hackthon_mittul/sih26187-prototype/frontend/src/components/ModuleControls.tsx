import React from 'react';
import { updateModules } from '../services/api';
import type { ModuleConfig } from '../services/api';

interface ModuleControlsProps {
  feedId: string | null;
  modules: ModuleConfig | null;
  isLive: boolean;
  onModulesChanged: () => void;
}

const ModuleControls: React.FC<ModuleControlsProps> = ({
  feedId,
  modules,
  isLive,
  onModulesChanged,
}) => {
  const handleToggle = async (key: keyof ModuleConfig) => {
    if (!feedId || !modules || !isLive) return;

    try {
      await updateModules(feedId, { [key]: !modules[key] });
      onModulesChanged();
    } catch (e) {
      console.error('Failed to toggle AI module:', e);
    }
  };

  const handleEnableAll = async () => {
    if (!feedId || !isLive) return;

    try {
      await updateModules(feedId, {
        human_detection: true,
        human_tracking: true,
        vehicle_detection: true,
        virtual_fence: true,
      });
      onModulesChanged();
    } catch (e) {
      console.error('Failed to enable all AI modules:', e);
    }
  };

  const moduleItems: { key: keyof ModuleConfig; label: string; icon: string; desc: string }[] = [
    {
      key: 'human_detection',
      label: 'Person Detection',
      icon: '👤',
      desc: 'YOLOv8 Real-time Human Detector',
    },
    {
      key: 'human_tracking',
      label: 'ByteTrack Multi-Object',
      icon: '🎯',
      desc: 'Persistent Track ID & Pathing',
    },
    {
      key: 'vehicle_detection',
      label: 'Vehicle Classification',
      icon: '🚗',
      desc: 'Cars, Bikes, Buses, Trucks',
    },
    {
      key: 'virtual_fence',
      label: 'Virtual Fence Intrusion',
      icon: '⚡',
      desc: 'Polygon Boundary Breach Radar',
    },
  ];

  return (
    <div className="panel command-panel" id="ai-module-controls-panel">
      <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className="panel-title">AI PIPELINE MODULES</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          {isLive && (
            <button
              className="cctv-btn primary compact"
              onClick={handleEnableAll}
              title="Engage all 4 AI pipeline modules"
              id="enable-all-modules-btn"
            >
              ⚡ ENABLE ALL
            </button>
          )}
          <span className={`status-indicator ${isLive ? 'success' : 'neutral'}`}>
            <span className="status-dot" />
            {isLive ? 'ACTIVE PIPELINE' : 'STANDBY'}
          </span>
        </div>
      </div>

      <div className="module-list">
        {moduleItems.map(({ key, label, icon, desc }) => {
          const enabled = modules ? modules[key] : false;
          const canToggle = isLive && feedId !== null && modules !== null;

          return (
            <div
              key={key}
              className={`module-card ${enabled ? 'active' : 'inactive'} ${canToggle ? 'interactive' : 'disabled'}`}
              onClick={() => canToggle && handleToggle(key)}
              title={canToggle ? `Click to toggle ${label}` : 'Start feed to toggle modules'}
              id={`module-toggle-${key}`}
            >
              <div className="module-icon-box">{icon}</div>
              <div className="module-info">
                <div className="module-title-row">
                  <span className="module-name">{label}</span>
                  <span className={`module-status-tag ${enabled ? 'on' : 'off'}`}>
                    {enabled ? 'ENGAGED' : 'DISABLED'}
                  </span>
                </div>
                <div className="module-desc">{desc}</div>
              </div>

              <div className={`module-switch-pill ${enabled ? 'checked' : ''}`}>
                <div className="switch-thumb" />
              </div>
            </div>
          );
        })}
      </div>

      {!isLive && (
        <div className="panel-sub-note">
          Surveillance feed must be LIVE to dynamically toggle AI pipeline modules.
        </div>
      )}
    </div>
  );
};

export default ModuleControls;
