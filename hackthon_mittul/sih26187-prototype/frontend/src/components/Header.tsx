import React, { useState, useEffect } from 'react';
import type { Feed } from '../services/api';

interface HeaderProps {
  activeFeed: Feed | null;
  isLive: boolean;
}

const Header: React.FC<HeaderProps> = ({ activeFeed, isLive }) => {
  const [currentTime, setCurrentTime] = useState<string>('');

  useEffect(() => {
    const updateClock = () => {
      const now = new Date();
      setCurrentTime(now.toLocaleTimeString('en-US', { hour12: false }) + ' UTC+' + ((-now.getTimezoneOffset() / 60) >= 0 ? '+' : '') + (-now.getTimezoneOffset() / 60));
    };
    updateClock();
    const interval = setInterval(updateClock, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="command-header" id="command-center-header">
      {/* Left Branding */}
      <div className="header-branding">
        <div className="header-logo-badge">IBVAP</div>
        <div className="header-titles">
          <h1 className="header-main-title">COMMAND CENTER SURVEILLANCE DASHBOARD</h1>
          <p className="header-sub-title">Intelligent Border Video Analytics Platform • Unified Security Console</p>
        </div>
      </div>

      {/* Center Operational Status */}
      <div className="header-center-stats">
        <div className="header-stat-item">
          <span className="stat-label">SYSTEM HEALTH</span>
          <span className="stat-value ok">
            <span className="pulse-dot green" />
            ONLINE
          </span>
        </div>

        <div className="header-stat-item">
          <span className="stat-label">ACTIVE SURVEILLANCE</span>
          <span className={`stat-value ${isLive ? 'live' : 'idle'}`}>
            <span className={`pulse-dot ${isLive ? 'red' : 'gray'}`} />
            {isLive ? 'STREAM ACTIVE' : 'STANDBY'}
          </span>
        </div>

        {activeFeed && (
          <div className="header-stat-item">
            <span className="stat-label">ASSIGNED FEED</span>
            <span className="stat-value highlight">{activeFeed.name}</span>
          </div>
        )}
      </div>

      {/* Right Clock & Operator Telemetry */}
      <div className="header-right-telemetry">
        <div className="header-clock-box">
          <span className="clock-label">SYSTEM CLOCK</span>
          <span className="clock-digits" id="header-system-clock">{currentTime || '00:00:00'}</span>
        </div>
      </div>
    </header>
  );
};

export default Header;
