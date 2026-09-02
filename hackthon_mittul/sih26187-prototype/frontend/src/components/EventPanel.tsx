import React, { useState } from 'react';
import { acknowledgeEvent } from '../services/api';
import type { LiveEvent } from '../services/api';

interface EventPanelProps {
  events: LiveEvent[];
  acknowledgedEvents: Set<string>;
  onAcknowledge: (eventId: string) => void;
}

const EventPanel: React.FC<EventPanelProps> = ({ events, acknowledgedEvents, onAcknowledge }) => {
  const [filter, setFilter] = useState<'All' | 'Intrusions' | 'Watchlist' | 'Tracking'>('All');

  const filteredEvents = events.filter(e => {
    if (filter === 'All') return true;
    if (filter === 'Intrusions') return e.type.startsWith('INTRUSION');
    if (filter === 'Watchlist') return e.type.startsWith('WATCHLIST') || e.type.startsWith('FACE_WATCHLIST');
    if (filter === 'Tracking') return e.type === 'NEW_TRACK' || e.type === 'TRACK_DISAPPEARED';
    return true;
  }).slice().reverse(); // Newest first

  const handleAcknowledge = async (eventId: string) => {
    try {
      await acknowledgeEvent(eventId);
      onAcknowledge(eventId);
    } catch (e) {
      console.error('Failed to acknowledge incident event:', e);
    }
  };

  const getEventCategoryStyle = (type: string, ack: boolean) => {
    if (type.startsWith('INTRUSION')) {
      return ack ? 'event-intrusion-ack' : 'event-intrusion';
    }
    if (type.startsWith('WATCHLIST') || type.startsWith('FACE_WATCHLIST')) {
      return ack ? 'event-watchlist-ack' : 'event-watchlist';
    }
    return 'event-tracking';
  };

  return (
    <div className="panel command-panel incident-log-panel" id="incident-events-panel">
      <div className="panel-header">
        <span className="panel-title">INCIDENT LOG & AUDIT TRAIL</span>
        {/* Category Filter Pills */}
        <div className="filter-pill-group">
          {(['All', 'Intrusions', 'Watchlist', 'Tracking'] as const).map(cat => (
            <button
              key={cat}
              className={`filter-pill ${filter === cat ? 'active' : ''}`}
              onClick={() => setFilter(cat)}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      <div className="scrollable-list incident-event-list">
        {filteredEvents.length === 0 ? (
          <div className="empty-state-banner">
            <span>No events recorded for current session filter.</span>
          </div>
        ) : (
          filteredEvents.map((ev, idx) => {
            const eventId = `${ev.timestamp}-${ev.type}-${ev.track_id || ''}`;
            const isAck = acknowledgedEvents.has(eventId);
            const isCriticalAlert = ev.type.startsWith('INTRUSION') || ev.type.startsWith('WATCHLIST') || ev.type.startsWith('FACE_WATCHLIST');

            return (
              <div
                key={`${eventId}-${idx}`}
                className={`incident-item-card ${getEventCategoryStyle(ev.type, isAck)}`}
              >
                <div className="incident-card-top">
                  <span className="incident-type-tag">{ev.type}</span>
                  <span className="incident-timestamp">{ev.timestamp.toFixed(2)}s</span>
                </div>

                <div className="incident-desc-row">
                  <span className="incident-desc">{ev.description}</span>
                  {isCriticalAlert && !isAck && (
                    <button
                      className="cctv-btn success compact ack-btn"
                      onClick={() => handleAcknowledge(eventId)}
                      title="Mark as operator acknowledged"
                    >
                      <span>✓ Ack</span>
                    </button>
                  )}
                  {isCriticalAlert && isAck && (
                    <span className="ack-status-tag">✓ Acknowledged</span>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default EventPanel;
