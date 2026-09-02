import React, { useState, useRef } from 'react';
import {
  enrollWatchlist,
  enableWatchlistRecord,
  disableWatchlistRecord,
  deleteWatchlistRecord,
} from '../services/api';
import type { WatchlistRecord } from '../services/api';

interface WatchlistPanelProps {
  records: WatchlistRecord[];
  onRecordsChanged: () => void;
}

const WatchlistPanel: React.FC<WatchlistPanelProps> = ({ records, onRecordsChanged }) => {
  const [showForm, setShowForm] = useState(false);
  const [name, setName] = useState('');
  const [status, setStatus] = useState<'WATCHLIST' | 'CRITICAL'>('WATCHLIST');
  const [enrolling, setEnrolling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleEnroll = async () => {
    const photo = fileRef.current?.files?.[0];
    if (!name.trim()) {
      setError('Enter person name');
      return;
    }
    if (!photo) {
      setError('Select reference biometric photo');
      return;
    }

    setEnrolling(true);
    setError(null);
    try {
      await enrollWatchlist(name.trim(), status, photo);
      setName('');
      setShowForm(false);
      if (fileRef.current) fileRef.current.value = '';
      onRecordsChanged();
    } catch (e: any) {
      setError(e.message || 'Enrollment failed');
    } finally {
      setEnrolling(false);
    }
  };

  const handleToggle = async (record: WatchlistRecord) => {
    try {
      if (record.enabled) {
        await disableWatchlistRecord(record.id);
      } else {
        await enableWatchlistRecord(record.id);
      }
      onRecordsChanged();
    } catch (e: any) {
      setError(e.message || 'Failed to toggle record');
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Permanently delete this biometric record?')) return;
    try {
      await deleteWatchlistRecord(id);
      onRecordsChanged();
    } catch (e: any) {
      setError(e.message || 'Failed to delete record');
    }
  };

  const formatDate = (ts?: number) => {
    if (!ts) return '--';
    return new Date(ts * 1000).toLocaleDateString() + ' ' + new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="panel command-panel" id="watchlist-management-panel">
      <div className="panel-header">
        <span className="panel-title">WATCHLIST ENROLLMENT ({records.length})</span>
        <button
          className="cctv-btn primary compact"
          onClick={() => { setShowForm(!showForm); setError(null); }}
          id="watchlist-enroll-toggle-btn"
        >
          <span className="btn-icon">{showForm ? '✕' : '+'}</span>
          <span>{showForm ? 'Cancel' : 'Enroll Subject'}</span>
        </button>
      </div>

      {showForm && (
        <div className="watchlist-enroll-box">
          <div className="enroll-field-group">
            <input
              type="text"
              placeholder="Subject Full Name / Alias"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="cctv-input"
              id="enroll-name-input"
            />
            <select
              value={status}
              onChange={(e) => setStatus(e.target.value as 'WATCHLIST' | 'CRITICAL')}
              className="cctv-select"
              id="enroll-status-select"
            >
              <option value="WATCHLIST">WATCHLIST (Amber Alert)</option>
              <option value="CRITICAL">CRITICAL (Red Alarm)</option>
            </select>
          </div>

          <div className="enroll-photo-row">
            <input
              ref={fileRef}
              type="file"
              accept="image/jpeg,image/png,image/jpg,.jpg,.jpeg,.png"
              className="cctv-file-input"
              id="enroll-photo-input"
            />
          </div>

          <button
            className="cctv-btn primary hero"
            onClick={handleEnroll}
            disabled={enrolling}
            id="enroll-submit-btn"
          >
            <span className="btn-icon">⚡</span>
            <span>{enrolling ? 'Extracting ArcFace 512D Embedding...' : 'Extract & Enroll Biometric Profile'}</span>
          </button>
        </div>
      )}

      {error && (
        <div className="feed-error-toast" style={{ margin: '0.4rem 0' }}>
          ⚠️ {error}
        </div>
      )}

      <div className="scrollable-list watchlist-records-deck">
        {records.length === 0 ? (
          <div className="empty-state-banner">
            <span>No subjects enrolled in facial watchlist database.</span>
          </div>
        ) : (
          records.map((rec) => (
            <div key={rec.id} className="watchlist-record-card">
              <div className="record-header">
                <span className="record-name">{rec.name}</span>
                <span className={`wl-status-badge ${rec.status.toLowerCase()}`}>
                  {rec.status}
                </span>
              </div>

              <div className="record-meta-row">
                <span className="record-date">{formatDate(rec.created)}</span>
                <span className={`record-state-pill ${rec.enabled ? 'active' : 'disabled'}`}>
                  {rec.enabled ? '● Active' : '○ Inactive'}
                </span>
              </div>

              <div className="record-actions-row">
                <button
                  className={`cctv-btn compact ${rec.enabled ? 'warning' : 'success'}`}
                  onClick={() => handleToggle(rec)}
                  title={rec.enabled ? 'Disable matching' : 'Enable matching'}
                >
                  <span>{rec.enabled ? 'Disable' : 'Enable'}</span>
                </button>
                <button
                  className="cctv-btn danger compact"
                  onClick={() => handleDelete(rec.id)}
                  title="Delete record from database"
                >
                  <span>Delete</span>
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default WatchlistPanel;
