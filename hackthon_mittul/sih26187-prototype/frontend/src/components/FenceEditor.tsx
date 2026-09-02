import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { saveZones } from '../services/api';
import type { Zone } from '../services/api';

interface FenceEditorProps {
  containerRef: React.RefObject<HTMLDivElement | null>;
  feedWidth: number;
  feedHeight: number;
  initialZone: Zone | null;
  onZoneSaved: () => void;
  isFenceEnabled: boolean;
  onToggleEnable: (enabled: boolean) => void;
  hasIntrusion?: boolean;
}

/**
 * Deep numeric coordinate comparison for polygons.
 * Avoids false negatives from object reference changes.
 */
function arePolygonsEqual(p1: number[][], p2: number[][]): boolean {
  if (p1.length !== p2.length) return false;
  for (let i = 0; i < p1.length; i++) {
    if (!p1[i] || !p2[i]) return false;
    if (Math.abs(p1[i][0] - p2[i][0]) > 0.5 || Math.abs(p1[i][1] - p2[i][1]) > 0.5) {
      return false;
    }
  }
  return true;
}

const FenceEditor: React.FC<FenceEditorProps> = ({
  containerRef,
  feedWidth,
  feedHeight,
  initialZone,
  onZoneSaved,
  isFenceEnabled,
  onToggleEnable,
  hasIntrusion = false,
}) => {
  // Mode state
  const [isEditing, setIsEditing] = useState(false);

  // SEPARATED STATE:
  // savedPolygon = polygon persisted in backend (authoritative)
  // editingPolygon = polygon currently being modified by the operator
  const [savedPolygon, setSavedPolygon] = useState<number[][]>(() =>
    initialZone?.polygon ? initialZone.polygon.map(pt => [...pt]) : []
  );
  const [editingPolygon, setEditingPolygon] = useState<number[][]>(() =>
    initialZone?.polygon ? initialZone.polygon.map(pt => [...pt]) : []
  );

  // Interaction state
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);
  const [isHoveringFirstPoint, setIsHoveringFirstPoint] = useState(false);
  const [draggingVertexIndex, setDraggingVertexIndex] = useState<number | null>(null);
  const hasDraggedRef = useRef(false);

  // Notification and UI state
  const [message, setMessage] = useState<{ text: string; type: 'success' | 'error' | 'info' } | null>(null);
  const [containerDims, setContainerDims] = useState<{ width: number; height: number }>({ width: 0, height: 0 });
  const [isSaving, setIsSaving] = useState(false);

  // Synchronize savedPolygon from initialZone only when NOT actively editing
  useEffect(() => {
    if (!isEditing) {
      const incoming = initialZone?.polygon ? initialZone.polygon.map(pt => [...pt]) : [];
      setSavedPolygon(incoming);
      setEditingPolygon(incoming.map(pt => [...pt]));
    }
  }, [initialZone, isEditing]);

  // Track container dimensions using ResizeObserver
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          setContainerDims({ width, height });
        }
      }
    });

    observer.observe(el);
    const rect = el.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      setContainerDims({ width: rect.width, height: rect.height });
    }

    return () => observer.disconnect();
  }, [containerRef]);

  // Global mouseup listener to finish vertex dragging even if mouse exits SVG
  useEffect(() => {
    const handleGlobalMouseUp = () => {
      if (draggingVertexIndex !== null) {
        setDraggingVertexIndex(null);
      }
    };
    window.addEventListener('mouseup', handleGlobalMouseUp);
    return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
  }, [draggingVertexIndex]);

  // Compute actual rendered video box within container (object-fit: contain)
  const videoGeometry = useMemo(() => {
    const { width: cw, height: ch } = containerDims;
    if (cw === 0 || ch === 0 || feedWidth <= 0 || feedHeight <= 0) {
      return null;
    }

    const videoRatio = feedWidth / feedHeight;
    const containerRatio = cw / ch;

    let renderedWidth: number;
    let renderedHeight: number;
    let offsetX: number;
    let offsetY: number;

    if (containerRatio > videoRatio) {
      // Pillarboxed (bars on left and right)
      renderedHeight = ch;
      renderedWidth = ch * videoRatio;
      offsetX = (cw - renderedWidth) / 2;
      offsetY = 0;
    } else {
      // Letterboxed (bars on top and bottom)
      renderedWidth = cw;
      renderedHeight = cw / videoRatio;
      offsetX = 0;
      offsetY = (ch - renderedHeight) / 2;
    }

    return {
      renderedWidth,
      renderedHeight,
      offsetX,
      offsetY,
      cw,
      ch,
    };
  }, [containerDims, feedWidth, feedHeight]);

  // Transform native video coordinate [x, y] to screen/SVG coordinate [sx, sy]
  const toScreen = useCallback((pt: number[]): [number, number] => {
    if (!videoGeometry || feedWidth <= 0 || feedHeight <= 0) return [0, 0];
    const sx = videoGeometry.offsetX + (pt[0] / feedWidth) * videoGeometry.renderedWidth;
    const sy = videoGeometry.offsetY + (pt[1] / feedHeight) * videoGeometry.renderedHeight;
    return [sx, sy];
  }, [videoGeometry, feedWidth, feedHeight]);

  // Active points to display: editingPolygon while in edit mode, savedPolygon otherwise
  const activePoints = useMemo(() => {
    return isEditing ? editingPolygon : savedPolygon;
  }, [isEditing, editingPolygon, savedPolygon]);

  // Deep comparison to track unsaved changes during edit mode
  const hasUnsavedChanges = useMemo(() => {
    if (!isEditing) return false;
    return !arePolygonsEqual(editingPolygon, savedPolygon);
  }, [isEditing, editingPolygon, savedPolygon]);

  // Screen SVG path string
  const svgPath = useMemo(() => {
    if (!videoGeometry || activePoints.length === 0) return '';
    return activePoints.map(pt => {
      const [sx, sy] = toScreen(pt);
      return `${sx.toFixed(1)},${sy.toFixed(1)}`;
    }).join(' ');
  }, [activePoints, videoGeometry, toScreen]);

  // Handle pointer movement over the SVG
  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!isEditing || !videoGeometry) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    setMousePos({ x: mx, y: my });

    // Handle vertex drag
    if (draggingVertexIndex !== null) {
      hasDraggedRef.current = true;
      const { offsetX, offsetY, renderedWidth, renderedHeight } = videoGeometry;
      const clampedX = Math.max(offsetX, Math.min(offsetX + renderedWidth, mx));
      const clampedY = Math.max(offsetY, Math.min(offsetY + renderedHeight, my));
      const normX = (clampedX - offsetX) / renderedWidth;
      const normY = (clampedY - offsetY) / renderedHeight;
      const nativeX = Math.round(Math.max(0, Math.min(feedWidth, normX * feedWidth)));
      const nativeY = Math.round(Math.max(0, Math.min(feedHeight, normY * feedHeight)));

      setEditingPolygon(prev => {
        const next = prev.map(pt => [...pt]);
        if (next[draggingVertexIndex]) {
          next[draggingVertexIndex] = [nativeX, nativeY];
        }
        return next;
      });
      return;
    }

    // Check if hovering near the first vertex (radius = 16px) to offer "click to close"
    if (editingPolygon.length >= 3) {
      const [firstSx, firstSy] = toScreen(editingPolygon[0]);
      const dist = Math.hypot(mx - firstSx, my - firstSy);
      setIsHoveringFirstPoint(dist <= 16);
    } else {
      setIsHoveringFirstPoint(false);
    }
  };

  const handleMouseLeave = () => {
    setMousePos(null);
    setIsHoveringFirstPoint(false);
  };

  // Convert click to native video coordinates & append point
  const handleSvgClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!isEditing) return;

    // If user just finished dragging a vertex handle, ignore the click
    if (hasDraggedRef.current) {
      hasDraggedRef.current = false;
      return;
    }

    if (draggingVertexIndex !== null) {
      setDraggingVertexIndex(null);
      return;
    }

    if (!videoGeometry || feedWidth <= 0 || feedHeight <= 0) {
      setMessage({ text: 'Video resolution unavailable. Cannot mark fence.', type: 'error' });
      setTimeout(() => setMessage(null), 3000);
      return;
    }

    // If clicking first point when >= 3 points exist, finish the polygon
    if (isHoveringFirstPoint && editingPolygon.length >= 3) {
      handleFinishDrawing();
      return;
    }

    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const { offsetX, offsetY, renderedWidth, renderedHeight } = videoGeometry;

    // Reject clicks outside the rendered video frame
    if (
      clickX < offsetX ||
      clickX > offsetX + renderedWidth ||
      clickY < offsetY ||
      clickY > offsetY + renderedHeight
    ) {
      return;
    }

    // Map to native video resolution
    const normX = (clickX - offsetX) / renderedWidth;
    const normY = (clickY - offsetY) / renderedHeight;

    const nativeX = Math.round(Math.max(0, Math.min(feedWidth, normX * feedWidth)));
    const nativeY = Math.round(Math.max(0, Math.min(feedHeight, normY * feedHeight)));

    setEditingPolygon(prev => [...prev, [nativeX, nativeY]]);
  };

  // Enter edit mode: clone savedPolygon into editingPolygon
  const handleStartEditing = () => {
    if (feedWidth <= 0 || feedHeight <= 0) {
      setMessage({ text: 'Select a valid feed before marking virtual fence', type: 'error' });
      setTimeout(() => setMessage(null), 3000);
      return;
    }
    setIsEditing(true);
    const cloned = savedPolygon.map(pt => [...pt]);
    setEditingPolygon(cloned);
    setDraggingVertexIndex(null);
    hasDraggedRef.current = false;

    if (cloned.length >= 3) {
      setMessage({ text: 'Editing existing fence. Drag vertices or click to add points.', type: 'info' });
    } else {
      setMessage({ text: 'Click on video to add perimeter points. Click #1 to close polygon.', type: 'info' });
    }
  };

  // Finish polygon drawing without exiting edit mode (retains active editing state for Save & Apply)
  const handleFinishDrawing = () => {
    if (editingPolygon.length < 3) {
      setMessage({ text: 'Polygon requires at least 3 points', type: 'error' });
      setTimeout(() => setMessage(null), 3000);
      return;
    }
    setMousePos(null);
    setIsHoveringFirstPoint(false);
    setMessage({ text: `Perimeter closed (${editingPolygon.length} vertices). Click "SAVE & APPLY" to commit.`, type: 'info' });
  };

  // Undo last point in editing polygon
  const handleUndoPoint = () => {
    if (editingPolygon.length > 0) {
      setEditingPolygon(prev => prev.slice(0, -1));
    }
  };

  // Clear editing polygon only (does not delete saved backend fence until saved)
  const handleClear = () => {
    setEditingPolygon([]);
    setDraggingVertexIndex(null);
    hasDraggedRef.current = false;
    setMessage({ text: 'Editor cleared. Draw new polygon or Cancel.', type: 'info' });
    setTimeout(() => setMessage(null), 2500);
  };

  // Cancel edit mode and revert editing state to saved polygon without saving
  const handleCancel = () => {
    const reverted = savedPolygon.map(pt => [...pt]);
    setEditingPolygon(reverted);
    setIsEditing(false);
    setDraggingVertexIndex(null);
    hasDraggedRef.current = false;
    setMousePos(null);
    setIsHoveringFirstPoint(false);
    setMessage({ text: 'Edit cancelled. Saved fence restored.', type: 'info' });
    setTimeout(() => setMessage(null), 2000);
  };

  // Save & Apply: sends CURRENT editingPolygon to backend
  const handleSave = async (enabledState: boolean = isFenceEnabled, pointsToSave: number[][] = editingPolygon) => {
    if (pointsToSave.length < 3) {
      setMessage({ text: 'Fence requires at least 3 vertices to save', type: 'error' });
      setTimeout(() => setMessage(null), 3000);
      return;
    }

    setIsSaving(true);
    try {
      const res = await saveZones('Restricted Border Zone', enabledState, pointsToSave);
      const cloned = pointsToSave.map(pt => [...pt]);
      setSavedPolygon(cloned);
      setEditingPolygon(cloned.map(pt => [...pt]));
      setIsEditing(false);
      setDraggingVertexIndex(null);
      hasDraggedRef.current = false;
      setMessage({ text: res.message || 'Fence Saved', type: 'success' });
      onZoneSaved();
      setTimeout(() => setMessage(null), 3500);
    } catch (e: any) {
      setMessage({ text: e.message || 'Failed to save virtual fence', type: 'error' });
      setTimeout(() => setMessage(null), 3500);
    } finally {
      setIsSaving(false);
    }
  };

  // Toggle Fence Enable/Disable
  const handleToggle = () => {
    const newState = !isFenceEnabled;
    onToggleEnable(newState);
    const targetPoints = isEditing && editingPolygon.length >= 3 ? editingPolygon : savedPolygon;
    if (targetPoints.length >= 3) {
      handleSave(newState, targetPoints);
    }
  };

  // Vertex mouse down for drag repositioning
  const handleVertexMouseDown = (e: React.MouseEvent, idx: number) => {
    e.stopPropagation();
    if (!isEditing) return;
    setDraggingVertexIndex(idx);
    hasDraggedRef.current = false;
  };

  const hasSavedFence = savedPolygon.length >= 3;
  const hasEditingFence = editingPolygon.length >= 3;

  const lastPoint = isEditing && editingPolygon.length > 0 ? editingPolygon[editingPolygon.length - 1] : null;
  const lastPointScreen = lastPoint ? toScreen(lastPoint) : null;
  const firstPointScreen = isEditing && editingPolygon.length > 0 ? toScreen(editingPolygon[0]) : null;

  // Compute status pill text
  const statusBadgeText = useMemo(() => {
    if (isEditing) {
      if (hasUnsavedChanges) return '⚠️ UNSAVED CHANGES';
      return '✏️ EDITING FENCE';
    }
    if (hasSavedFence) {
      return isFenceEnabled
        ? `🛡️ FENCE ACTIVE (${savedPolygon.length} PTS)`
        : `⏸️ FENCE DISABLED (${savedPolygon.length} PTS)`;
    }
    return 'NO FENCE CONFIGURED';
  }, [isEditing, hasUnsavedChanges, hasSavedFence, isFenceEnabled, savedPolygon.length]);

  return (
    <>
      {/* Top Status Pill on Video Canvas */}
      <div
        className={`fence-state-pill ${
          isEditing
            ? hasUnsavedChanges
              ? 'unsaved'
              : 'editing'
            : hasSavedFence
              ? isFenceEnabled
                ? 'active'
                : 'disabled'
              : 'empty'
        }`}
      >
        <span className="pill-dot" />
        <span className="pill-label">{statusBadgeText}</span>
      </div>

      {/* SVG Layer for Drawing, Editing & Visual Overlay */}
      {(isEditing || isFenceEnabled || activePoints.length > 0) && (
        <svg
          className="fence-svg-overlay"
          onClick={handleSvgClick}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            zIndex: 12,
            pointerEvents: isEditing ? 'auto' : 'none',
            cursor: isEditing
              ? draggingVertexIndex !== null
                ? 'grabbing'
                : isHoveringFirstPoint
                  ? 'pointer'
                  : 'crosshair'
              : 'default',
          }}
        >
          <defs>
            <filter id="fenceGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* Rendered Polygon */}
          {svgPath && (
            <polygon
              points={svgPath}
              fill={
                isEditing
                  ? hasUnsavedChanges
                    ? 'rgba(245, 158, 11, 0.18)'
                    : 'rgba(6, 182, 212, 0.18)'
                  : hasIntrusion
                    ? 'rgba(239, 68, 68, 0.28)'
                    : isFenceEnabled
                      ? 'rgba(16, 185, 129, 0.14)'
                      : 'rgba(100, 116, 139, 0.08)'
              }
              stroke={
                isEditing
                  ? hasUnsavedChanges
                    ? '#f59e0b'
                    : '#06b6d4'
                  : hasIntrusion
                    ? '#ef4444'
                    : isFenceEnabled
                      ? '#10b981'
                      : '#64748b'
              }
              strokeWidth={isEditing ? 2.5 : 2.5}
              strokeDasharray={isEditing ? '6,4' : 'none'}
              filter={hasIntrusion ? 'url(#fenceGlow)' : undefined}
            />
          )}

          {/* Rubber-band connecting line to mouse pointer during editing */}
          {isEditing && lastPointScreen && mousePos && draggingVertexIndex === null && (
            <line
              x1={lastPointScreen[0]}
              y1={lastPointScreen[1]}
              x2={isHoveringFirstPoint && firstPointScreen ? firstPointScreen[0] : mousePos.x}
              y2={isHoveringFirstPoint && firstPointScreen ? firstPointScreen[1] : mousePos.y}
              stroke={isHoveringFirstPoint ? '#10b981' : '#06b6d4'}
              strokeWidth={isHoveringFirstPoint ? 2.5 : 1.5}
              strokeDasharray="4,4"
            />
          )}

          {/* Vertex handles and numbers during editing */}
          {isEditing && editingPolygon.map((pt, idx) => {
            const [sx, sy] = toScreen(pt);
            const isFirst = idx === 0;
            const isCloseTarget = isFirst && isHoveringFirstPoint && editingPolygon.length >= 3;
            const isDragging = draggingVertexIndex === idx;

            return (
              <g
                key={idx}
                className="fence-vertex-handle"
                onMouseDown={(e) => handleVertexMouseDown(e, idx)}
                style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
              >
                {/* Outer touch/drag area circle */}
                <circle
                  cx={sx}
                  cy={sy}
                  r={isDragging ? 12 : isCloseTarget ? 10 : isFirst ? 8 : 6.5}
                  fill={isDragging ? '#ec4899' : isCloseTarget ? '#10b981' : isFirst ? '#f59e0b' : '#06b6d4'}
                  stroke="#080b11"
                  strokeWidth="2.5"
                />
                {/* Vertex index label */}
                <text
                  x={sx}
                  y={sy - 11}
                  fill="#ffffff"
                  fontSize="10"
                  fontFamily="'JetBrains Mono', monospace"
                  fontWeight="700"
                  textAnchor="middle"
                  style={{
                    textShadow: '0 0 4px #000, 0 0 2px #000',
                    pointerEvents: 'none',
                    userSelect: 'none',
                  }}
                >
                  {isCloseTarget ? 'CLOSE' : `#${idx + 1}`}
                </text>
              </g>
            );
          })}
        </svg>
      )}

      {/* Floating Notification Message */}
      {message && (
        <div
          className={`fence-status-toast ${message.type}`}
          style={{
            position: 'absolute',
            top: '2.5rem',
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 30,
            padding: '0.4rem 0.85rem',
            borderRadius: '4px',
            fontSize: '0.75rem',
            fontWeight: 600,
            backdropFilter: 'blur(6px)',
            pointerEvents: 'none',
            display: 'flex',
            alignItems: 'center',
            gap: '0.45rem',
            boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
            border: `1px solid ${
              message.type === 'error'
                ? '#ef4444'
                : message.type === 'success'
                  ? '#10b981'
                  : '#06b6d4'
            }`,
            background: 'rgba(14, 19, 31, 0.95)',
            color:
              message.type === 'error'
                ? '#ef4444'
                : message.type === 'success'
                  ? '#10b981'
                  : '#38bdf8',
          }}
        >
          <span>{message.type === 'error' ? '⚠️' : message.type === 'success' ? '✓' : 'ℹ️'}</span>
          <span>{message.text}</span>
        </div>
      )}

      {/* Fence Action Controls Bar */}
      <div className="fence-toolbar-overlay">
        {!isEditing ? (
          <div className="fence-btn-group">
            <button
              className="cctv-btn primary"
              onClick={handleStartEditing}
              title={hasSavedFence ? 'Edit and modify existing virtual fence' : 'Draw new polygon fence boundaries'}
              id="cctv-fence-edit-btn"
            >
              <span className="btn-icon">✏️</span>
              <span>{hasSavedFence ? 'Edit Fence' : 'Draw Fence'}</span>
            </button>
            {hasSavedFence && (
              <button
                className={`cctv-btn ${isFenceEnabled ? 'warning' : 'success'}`}
                onClick={handleToggle}
                title={isFenceEnabled ? 'Disable virtual fence detection' : 'Enable virtual fence detection'}
                id="cctv-fence-toggle-btn"
              >
                <span className="btn-icon">{isFenceEnabled ? '⏸' : '▶'}</span>
                <span>{isFenceEnabled ? 'Disable' : 'Enable'}</span>
              </button>
            )}
          </div>
        ) : (
          <div className="fence-btn-group editing">
            <button
              className="cctv-btn success"
              onClick={handleFinishDrawing}
              disabled={editingPolygon.length < 3}
              title="Close perimeter polygon"
              id="cctv-fence-finish-btn"
            >
              <span className="btn-icon">✓</span>
              <span>Finish ({editingPolygon.length})</span>
            </button>
            <button
              className="cctv-btn"
              onClick={handleUndoPoint}
              disabled={editingPolygon.length === 0}
              title="Remove last placed point"
              id="cctv-fence-undo-btn"
            >
              <span className="btn-icon">↩</span>
              <span>Undo</span>
            </button>
            <button
              className="cctv-btn danger"
              onClick={handleClear}
              disabled={editingPolygon.length === 0}
              title="Clear all points in editor to redraw"
              id="cctv-fence-clear-btn"
            >
              <span className="btn-icon">🗑</span>
              <span>Clear</span>
            </button>
            <button
              className="cctv-btn"
              onClick={handleCancel}
              title="Cancel drawing and revert to saved fence"
              id="cctv-fence-cancel-btn"
            >
              <span className="btn-icon">✕</span>
              <span>Cancel</span>
            </button>
            <button
              className={`cctv-btn ${hasUnsavedChanges ? 'primary active-pulse' : 'primary'}`}
              onClick={() => handleSave(true, editingPolygon)}
              disabled={!hasEditingFence || isSaving}
              title="Save fence and apply immediately to live detection"
              id="cctv-fence-save-btn"
            >
              <span className="btn-icon">💾</span>
              <span>{isSaving ? 'Saving...' : 'Save & Apply'}</span>
            </button>
          </div>
        )}
      </div>
    </>
  );
};

export default FenceEditor;
