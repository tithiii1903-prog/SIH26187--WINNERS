import React, { useEffect, useState, useCallback, useRef } from 'react';
import Header from '../components/Header';
import CCTVViewer from '../components/CCTVViewer';
import type { CCTVPlayerStatus } from '../components/CCTVViewer';
import HDFaceViewer from '../components/HDFaceViewer';
import PersonAnalytics from '../components/PersonAnalytics';
import VehicleAnalytics from '../components/VehicleAnalytics';
import IntrusionPanel from '../components/IntrusionPanel';
import EventPanel from '../components/EventPanel';
import FeedManager from '../components/FeedManager';
import ModuleControls from '../components/ModuleControls';
import WatchlistPanel from '../components/WatchlistPanel';
import WatchlistAlert from '../components/WatchlistAlert';
import {
  fetchFeeds,
  fetchFeedAnalytics,
  fetchFeedEvents,
  startFeed,
  stopFeed,
  fetchZones,
  fetchAcknowledgedEvents,
  fetchWatchlist,
  fetchFaceCameraStatus,
  fetchFaceCameraResults,
  startFaceCamera,
  stopFaceCamera,
} from '../services/api';
import type {
  Feed,
  LiveAnalytics,
  LiveEvent,
  Zone,
  WatchlistRecord,
  FaceCameraStatus,
  FaceCameraResults,
} from '../services/api';

const Dashboard: React.FC = () => {
  // Feed registry state
  const [feeds, setFeeds] = useState<Feed[]>([]);
  const [selectedFeedId, setSelectedFeedId] = useState<string | null>(null);

  // Authoritative Feed Player State Machine
  const [playerStatus, setPlayerStatus] = useState<CCTVPlayerStatus>('IDLE');
  const [streamSessionId, setStreamSessionId] = useState<number | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Live analytics & incidents from backend
  const [analytics, setAnalytics] = useState<LiveAnalytics | null>(null);
  const [events, setEvents] = useState<LiveEvent[]>([]);

  // HD Face Camera subsystem state (Phase 13)
  const [faceCameraStatus, setFaceCameraStatus] = useState<FaceCameraStatus | null>(null);
  const [faceCameraResults, setFaceCameraResults] = useState<FaceCameraResults | null>(null);

  // Zone / Fence state
  const [zones, setZones] = useState<Zone[]>([]);
  const [acknowledgedEvents, setAcknowledgedEvents] = useState<Set<string>>(new Set());

  // Watchlist state
  const [watchlistRecords, setWatchlistRecords] = useState<WatchlistRecord[]>([]);

  // Transition & Polling Refs
  const isTransitioningRef = useRef<boolean>(false);
  const feedPollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const facePollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Derived selected feed
  const selectedFeed = feeds.find(f => f.id === selectedFeedId) || null;
  const isLive = playerStatus === 'LIVE';

  // Load Feeds
  const loadFeeds = useCallback(async () => {
    try {
      const feedList = await fetchFeeds();
      setFeeds(feedList);
      return feedList;
    } catch (err) {
      console.error('Failed to load feeds:', err);
      return [];
    }
  }, []);

  // Load Zones
  const loadZones = useCallback(async () => {
    try {
      const zData = await fetchZones();
      setZones(zData.zones || []);
    } catch (err) {
      console.error('Failed to load zones:', err);
    }
  }, []);

  // Load Watchlist
  const loadWatchlist = useCallback(async () => {
    try {
      const records = await fetchWatchlist();
      setWatchlistRecords(records);
    } catch (err) {
      console.error('Failed to load watchlist:', err);
    }
  }, []);

  // Load HD Face Camera Status
  const loadFaceStatus = useCallback(async () => {
    try {
      const st = await fetchFaceCameraStatus();
      setFaceCameraStatus(st);
      if (st.is_running) {
        const res = await fetchFaceCameraResults();
        setFaceCameraResults(res);
      } else {
        setFaceCameraResults(null);
      }
    } catch (err) {
      // silent catch for polling
    }
  }, []);

  // Initial Data Load
  useEffect(() => {
    loadFeeds().then((feedList) => {
      if (feedList.length > 0 && !selectedFeedId) {
        // Auto-select first feed if available
        setSelectedFeedId(feedList[0].id);
      }
    });
    loadZones();
    loadWatchlist();
    loadFaceStatus();
    fetchAcknowledgedEvents()
      .then(data => setAcknowledgedEvents(new Set(data.acknowledged_events || [])))
      .catch(console.error);

    return () => {
      if (feedPollingRef.current) clearInterval(feedPollingRef.current);
      if (facePollingRef.current) clearInterval(facePollingRef.current);
    };
  }, [loadFeeds, loadZones, loadWatchlist, loadFaceStatus]);

  // Synchronize Player Status when selected feed changes or is initialized
  useEffect(() => {
    if (!selectedFeed) {
      setPlayerStatus('IDLE');
      setStreamSessionId(null);
      setAnalytics(null);
      return;
    }

    if (isTransitioningRef.current) return;

    if (selectedFeed.status === 'LIVE') {
      setPlayerStatus('LIVE');
      setStreamSessionId(prev => prev ?? Date.now());
    } else if (selectedFeed.status === 'STARTING') {
      setPlayerStatus('STARTING');
    } else if (selectedFeed.status === 'ERROR') {
      setPlayerStatus('ERROR');
      setStreamSessionId(null);
    } else if (selectedFeed.status === 'STOPPED') {
      setPlayerStatus('STOPPED');
      setStreamSessionId(null);
    } else {
      setPlayerStatus('READY');
      setStreamSessionId(null);
    }
  }, [selectedFeedId, selectedFeed?.status]);

  // Feed Polling Loop: Active only when feed is LIVE or STARTING
  useEffect(() => {
    if (feedPollingRef.current) {
      clearInterval(feedPollingRef.current);
      feedPollingRef.current = null;
    }

    if (!selectedFeedId || (playerStatus !== 'LIVE' && playerStatus !== 'STARTING')) {
      return;
    }

    const currentFeedId = selectedFeedId;

    const poll = async () => {
      try {
        const feedList = await fetchFeeds();
        setFeeds(feedList);

        const current = feedList.find(f => f.id === currentFeedId);
        if (!current) return;

        if (current.status === 'LIVE') {
          if (playerStatus !== 'LIVE') {
            setPlayerStatus('LIVE');
            setStreamSessionId(prev => prev ?? Date.now());
          }
          const [analyticsData, eventsData] = await Promise.all([
            fetchFeedAnalytics(currentFeedId),
            fetchFeedEvents(currentFeedId),
          ]);
          setAnalytics(analyticsData);
          setEvents(eventsData);
        } else if (current.status === 'STOPPED') {
          // Backend finished or hit EOF
          setPlayerStatus('STOPPED');
          setStreamSessionId(null);
          setAnalytics(null);
        } else if (current.status === 'ERROR') {
          setPlayerStatus('ERROR');
          setStreamSessionId(null);
          setAnalytics(null);
        }
      } catch (err) {
        console.error('Feed polling error:', err);
      }
    };

    poll();
    feedPollingRef.current = setInterval(poll, 1000);

    return () => {
      if (feedPollingRef.current) {
        clearInterval(feedPollingRef.current);
        feedPollingRef.current = null;
      }
    };
  }, [selectedFeedId, playerStatus]);

  // HD Face Camera Polling Loop
  useEffect(() => {
    if (facePollingRef.current) {
      clearInterval(facePollingRef.current);
      facePollingRef.current = null;
    }

    const pollFace = async () => {
      try {
        const st = await fetchFaceCameraStatus();
        setFaceCameraStatus(st);
        if (st.is_running) {
          const res = await fetchFaceCameraResults();
          setFaceCameraResults(res);
        } else {
          setFaceCameraResults(null);
        }
      } catch (err) {
        // silent catch
      }
    };

    pollFace();
    facePollingRef.current = setInterval(pollFace, 1000);

    return () => {
      if (facePollingRef.current) {
        clearInterval(facePollingRef.current);
        facePollingRef.current = null;
      }
    };
  }, []);

  // START Feed Handler with transition guard
  const handleStart = async () => {
    if (!selectedFeedId || isTransitioningRef.current) return;

    isTransitioningRef.current = true;
    setErrorMessage(null);
    setPlayerStatus('STARTING');

    try {
      await startFeed(selectedFeedId);
      const sessionKey = Date.now();
      setStreamSessionId(sessionKey);

      // Verify backend transition
      const updatedList = await loadFeeds();
      const current = updatedList.find(f => f.id === selectedFeedId);
      if (current && current.status === 'LIVE') {
        setPlayerStatus('LIVE');
      } else {
        // Will be picked up by polling loop
        setPlayerStatus('STARTING');
      }
    } catch (err: any) {
      console.error('Start feed error:', err);
      setErrorMessage(err.message || 'Unable to start surveillance feed');
      setPlayerStatus('ERROR');
      setStreamSessionId(null);
    } finally {
      isTransitioningRef.current = false;
    }
  };

  // STOP Feed Handler with transition guard
  const handleStop = async () => {
    if (!selectedFeedId || isTransitioningRef.current) return;

    isTransitioningRef.current = true;
    setPlayerStatus('STOPPING');
    setStreamSessionId(null); // Immediately unmount MJPEG stream

    try {
      await stopFeed(selectedFeedId);
      await loadFeeds();
      setPlayerStatus('STOPPED');
      setAnalytics(null);
    } catch (err: any) {
      console.error('Stop feed error:', err);
      setErrorMessage(err.message || 'Failed to stop feed');
      setPlayerStatus('ERROR');
    } finally {
      isTransitioningRef.current = false;
    }
  };

  // REPLAY Feed Handler (clean unmount -> start from frame 0)
  const handleReplay = async () => {
    if (!selectedFeedId || isTransitioningRef.current) return;

    isTransitioningRef.current = true;
    setErrorMessage(null);
    setStreamSessionId(null); // Ensure previous stream element is unmounted
    setPlayerStatus('STARTING');

    try {
      // If currently live, stop it first
      if (selectedFeed?.status === 'LIVE') {
        await stopFeed(selectedFeedId).catch(() => {});
      }

      await startFeed(selectedFeedId);
      const freshSessionKey = Date.now();
      setStreamSessionId(freshSessionKey);

      await loadFeeds();
      setPlayerStatus('LIVE');
    } catch (err: any) {
      console.error('Replay feed error:', err);
      setErrorMessage(err.message || 'Replay failed to initialize');
      setPlayerStatus('ERROR');
      setStreamSessionId(null);
    } finally {
      isTransitioningRef.current = false;
    }
  };

  // Feed selection change
  const handleSelectFeed = (feedId: string) => {
    if (selectedFeedId === feedId) return;
    if (isLive) {
      // Prompt or automatically stop
      handleStop().then(() => {
        setSelectedFeedId(feedId);
      });
    } else {
      setSelectedFeedId(feedId);
    }
  };

  // HD Face Camera Handlers
  const handleStartFaceCamera = async (deviceIndex: number = 0) => {
    try {
      await startFaceCamera(deviceIndex);
      await loadFaceStatus();
    } catch (err: any) {
      console.error('Start HD face camera error:', err);
    }
  };

  const handleStopFaceCamera = async () => {
    try {
      await stopFaceCamera();
      await loadFaceStatus();
    } catch (err: any) {
      console.error('Stop HD face camera error:', err);
    }
  };

  // Acknowledgement Handler
  const handleAcknowledge = (eventId: string) => {
    setAcknowledgedEvents(prev => new Set([...prev, eventId]));
  };

  const handleZoneSaved = () => {
    loadZones();
  };

  const activeZone = zones.find(z => z.id === 'restricted-border-zone') || (zones.length > 0 ? zones[0] : null);
  const isFenceEnabled = activeZone ? (activeZone.enabled ?? true) : true;
  const hasIntrusions = Boolean(
    analytics && isLive && analytics.active_intrusions && analytics.active_intrusions.length > 0
  );

  return (
    <div className="dashboard-container command-center-layout">
      {/* Top Command Center Header */}
      <Header activeFeed={selectedFeed} isLive={isLive} />

      {/* Main Command Room Workspace */}
      <main className="command-center-workspace">
        {/* Left / Center: Surveillance & Telemetry Deck */}
        <section className="surveillance-deck">
          {/* Primary CCTV Viewport with Integrated Control Bar & HUD */}
          <CCTVViewer
            feed={selectedFeed}
            status={playerStatus}
            isLive={isLive}
            streamSessionId={streamSessionId}
            onStart={handleStart}
            onStop={handleStop}
            onReplay={handleReplay}
            initialZone={activeZone}
            onZoneSaved={handleZoneSaved}
            isFenceEnabled={isFenceEnabled}
            onToggleFenceEnable={() => {}}
            hasIntrusions={hasIntrusions}
            analytics={analytics}
            errorMessage={errorMessage}
          />

          {/* Subsystem Deck: HD Face Recognition & Analytics */}
          <div className="subsystem-grid">
            <div className="subsystem-col hd-face-col">
              <HDFaceViewer
                status={faceCameraStatus}
                results={faceCameraResults}
                onStart={handleStartFaceCamera}
                onStop={handleStopFaceCamera}
              />
            </div>
            <div className="subsystem-col analytics-col">
              <div className="analytics-metrics-stack">
                <PersonAnalytics
                  currentPersons={analytics && isLive ? analytics.current_persons : '--'}
                  peakPersons={analytics && isLive ? analytics.peak_persons : '--'}
                  activeTracks={analytics && isLive ? analytics.active_tracks : '--'}
                />
                <VehicleAnalytics
                  currentVehicles={analytics && isLive ? analytics.current_vehicles : '--'}
                  peakVehicles={analytics && isLive ? analytics.peak_vehicles : '--'}
                  cars={analytics && isLive ? analytics.cars : '--'}
                  motorcycles={analytics && isLive ? analytics.motorcycles : '--'}
                  buses={analytics && isLive ? analytics.buses : '--'}
                  trucks={analytics && isLive ? analytics.trucks : '--'}
                />
              </div>
            </div>
          </div>
        </section>

        {/* Right: Operator Control & Incident Sidebar */}
        <aside className="operator-sidebar">
          <FeedManager
            feeds={feeds}
            selectedFeedId={selectedFeedId}
            onSelectFeed={handleSelectFeed}
            onFeedsChanged={loadFeeds}
          />

          <ModuleControls
            feedId={selectedFeedId}
            modules={analytics?.modules || null}
            isLive={isLive}
            onModulesChanged={loadFeeds}
          />

          <IntrusionPanel activeIntrusions={analytics && isLive ? (analytics.active_intrusions || []) : []} />

          <WatchlistAlert faceMatches={faceCameraResults?.faces || []} />

          <WatchlistPanel
            records={watchlistRecords}
            onRecordsChanged={loadWatchlist}
          />

          <EventPanel
            events={events}
            acknowledgedEvents={acknowledgedEvents}
            onAcknowledge={handleAcknowledge}
          />
        </aside>
      </main>
    </div>
  );
};

export default Dashboard;
