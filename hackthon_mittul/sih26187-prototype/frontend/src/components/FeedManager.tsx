import React, { useState, useRef } from 'react';
import { uploadFeed, deleteFeed, createCameraFeed } from '../services/api';
import type { Feed } from '../services/api';

interface FeedManagerProps {
  feeds: Feed[];
  selectedFeedId: string | null;
  onSelectFeed: (feedId: string) => void;
  onFeedsChanged: () => void;
}

const FeedManager: React.FC<FeedManagerProps> = ({
  feeds,
  selectedFeedId,
  onSelectFeed,
  onFeedsChanged,
}) => {
  const [activeTab, setActiveTab] = useState<'recorded' | 'camera'>('recorded');
  const [showUpload, setShowUpload] = useState(false);
  const [showCameraAdd, setShowCameraAdd] = useState(false);
  const [feedName, setFeedName] = useState('');
  const [cameraName, setCameraName] = useState('');
  const [deviceIndex, setDeviceIndex] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [registeringCamera, setRegisteringCamera] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const recordedFeeds = feeds.filter(f => (f.source_type || 'file') === 'file');
  const cameraFeeds = feeds.filter(f => f.source_type === 'camera');
  const displayedFeeds = activeTab === 'recorded' ? recordedFeeds : cameraFeeds;

  const handleUpload = async () => {
    const file = fileInputRef.current?.files?.[0];
    if (!file) {
      setError('Select a video file to upload');
      return;
    }
    if (!feedName.trim()) {
      setError('Enter a feed identifier name');
      return;
    }

    setUploading(true);
    setError(null);
    try {
      await uploadFeed(file, feedName.trim());
      setFeedName('');
      setShowUpload(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
      onFeedsChanged();
    } catch (e: any) {
      setError(e.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleAddCamera = async () => {
    if (!cameraName.trim()) {
      setError('Enter a camera identifier name');
      return;
    }

    setRegisteringCamera(true);
    setError(null);
    try {
      await createCameraFeed(cameraName.trim(), deviceIndex);
      setCameraName('');
      setDeviceIndex(0);
      setShowCameraAdd(false);
      onFeedsChanged();
    } catch (e: any) {
      setError(e.message || 'Failed to connect device camera');
    } finally {
      setRegisteringCamera(false);
    }
  };

  const handleDelete = async (feedId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Permanently delete this surveillance source?')) return;
    try {
      await deleteFeed(feedId);
      onFeedsChanged();
    } catch (err: any) {
      setError(err.message || 'Delete failed');
    }
  };

  const handleQuickAddCamera = async () => {
    setRegisteringCamera(true);
    setError(null);
    try {
      const feed = await createCameraFeed("My WebCam Device", 0);
      onFeedsChanged();
      if (feed && feed.id) {
        onSelectFeed(feed.id);
      }
    } catch (e: any) {
      setError(e.message || 'Failed to connect device camera');
    } finally {
      setRegisteringCamera(false);
    }
  };

  return (
    <div className="panel command-panel" id="surveillance-sources-panel">
      <div className="panel-header">
        <span className="panel-title">SURVEILLANCE SOURCES ({displayedFeeds.length})</span>
        <div className="source-nav-tabs">
          <button
            className={`source-nav-tab ${activeTab === 'recorded' ? 'active' : ''}`}
            onClick={() => { setActiveTab('recorded'); setError(null); }}
          >
            📁 MP4 Sources
          </button>
          <button
            className={`source-nav-tab ${activeTab === 'camera' ? 'active' : ''}`}
            onClick={() => { setActiveTab('camera'); setError(null); }}
          >
            📷 Live Cams
          </button>
        </div>
      </div>

      {/* Action Add Button */}
      <div className="source-action-bar">
        {activeTab === 'recorded' ? (
          <button
            className="cctv-btn primary compact full-width"
            onClick={() => { setShowUpload(!showUpload); setShowCameraAdd(false); setError(null); }}
            id="sources-upload-btn"
          >
            <span className="btn-icon">{showUpload ? '✕' : '+'}</span>
            <span>{showUpload ? 'Cancel' : 'Upload MP4 Video'}</span>
          </button>
        ) : (
          <button
            className="cctv-btn primary compact full-width"
            onClick={() => { setShowCameraAdd(!showCameraAdd); setShowUpload(false); setError(null); }}
            id="sources-add-cam-btn"
          >
            <span className="btn-icon">{showCameraAdd ? '✕' : '+'}</span>
            <span>{showCameraAdd ? 'Cancel' : 'Register Device Camera'}</span>
          </button>
        )}
      </div>

      {/* Upload MP4 Form */}
      {showUpload && activeTab === 'recorded' && (
        <div className="source-form-card">
          <input
            type="text"
            placeholder="Feed Name (e.g. BORDER-WEST-CAM-01)"
            value={feedName}
            onChange={(e) => setFeedName(e.target.value)}
            disabled={uploading}
            className="source-input"
          />
          <input
            type="file"
            accept="video/mp4,video/x-m4v,video/*"
            ref={fileInputRef}
            disabled={uploading}
            className="source-file-input"
          />
          <button
            className="cctv-btn success compact full-width"
            onClick={handleUpload}
            disabled={uploading}
          >
            {uploading ? 'Uploading Video...' : 'Upload Feed'}
          </button>
        </div>
      )}

      {/* Register Camera Form */}
      {showCameraAdd && activeTab === 'camera' && (
        <div className="source-form-card">
          <input
            type="text"
            placeholder="Camera Name (e.g. PERIMETER-GATE-01)"
            value={cameraName}
            onChange={(e) => setCameraName(e.target.value)}
            disabled={registeringCamera}
            className="source-input"
          />
          <label className="source-dev-label">
            Device Index:
            <input
              type="number"
              min="0"
              max="10"
              value={deviceIndex}
              onChange={(e) => setDeviceIndex(parseInt(e.target.value) || 0)}
              disabled={registeringCamera}
              className="source-dev-input"
            />
          </label>
          <button
            className="cctv-btn success compact full-width"
            onClick={handleAddCamera}
            disabled={registeringCamera}
          >
            {registeringCamera ? 'Connecting Device...' : 'Connect Camera'}
          </button>
        </div>
      )}

      {error && (
        <div className="feed-error-toast" style={{ margin: '0.4rem 0' }}>
          ⚠️ {error}
        </div>
      )}

      {/* Feed List */}
      <div className="scrollable-list source-list-deck">
        {displayedFeeds.length === 0 ? (
          <div className="empty-state-banner" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', alignItems: 'center' }}>
            <span>{activeTab === 'recorded' ? 'No recorded feeds registered. Upload an MP4 video.' : 'No device cameras registered yet.'}</span>
            {activeTab === 'camera' && (
              <button
                className="cctv-btn primary compact"
                onClick={handleQuickAddCamera}
                disabled={registeringCamera}
                style={{ marginTop: '0.2rem' }}
              >
                📷 Quick Connect My WebCam
              </button>
            )}
          </div>
        ) : (
          displayedFeeds.map((f) => {
            const isSelected = selectedFeedId === f.id;
            const isRunning = f.status === 'LIVE';

            return (
              <div
                key={f.id}
                className={`source-feed-card ${isSelected ? 'selected' : ''}`}
                onClick={() => onSelectFeed(f.id)}
                id={`feed-item-${f.id}`}
              >
                <div className="source-feed-header">
                  <span className="source-icon">{(f.source_type || 'file') === 'camera' ? '📷' : '🎥'}</span>
                  <span className="source-feed-name">{f.name}</span>
                  <span className={`cctv-status-pill compact ${f.status.toLowerCase()}`}>
                    <span className="status-dot" />
                    <span>{f.status}</span>
                  </span>
                </div>

                <div className="source-feed-specs">
                  <span>{f.filename}</span>
                  <span>•</span>
                  <span>{f.width}×{f.height}</span>
                  <span>•</span>
                  <span>{f.fps} FPS</span>
                  {f.duration && (
                    <>
                      <span>•</span>
                      <span>{f.duration.toFixed(1)}s</span>
                    </>
                  )}
                </div>

                {!isRunning && f.status !== 'STARTING' && (
                  <button
                    className="source-delete-btn"
                    onClick={(e) => handleDelete(f.id, e)}
                    title="Delete feed"
                  >
                    ✕
                  </button>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default FeedManager;
