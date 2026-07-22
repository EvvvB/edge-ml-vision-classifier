import { Fragment, useEffect, useMemo, useState } from 'react'
import { useInfiniteQuery, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  apiFetch,
  deleteDetection,
  deleteDetections,
  detectionImageUrl,
  exportDownloadUrl,
} from './api.js'
import DeviceManager from './DeviceManager.jsx'
import DevicesPanel, { isPositioning, presenceOf } from './DevicesPanel.jsx'
import EvalView from './EvalPanel.jsx'
import FilterSidebar from './Filters.jsx'
import ReceiptsPanel from './ReceiptsPanel.jsx'
import TileSimulator, { inferenceRois } from './TileSimulator.jsx'
import { filtersFromUrl, syncFiltersToUrl } from './filterState.js'
import { isLocalEnvironment } from './env.js'

const PAGE_SIZE = 24
const POLL_INTERVAL_MS = 30_000
const DEVICES_POLL_INTERVAL_MS = 10_000

// datetime-local values are local wall-clock time; the API wants instants,
// so convert through Date to a UTC ISO string.
function isoParam(value) {
  if (!value) return undefined
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? undefined : date.toISOString()
}

function filterParams(filters) {
  return {
    device_id: filters.deviceId || undefined,
    labels: filters.labels.length > 0 ? filters.labels.join(',') : undefined,
    models: filters.models.length > 0 ? filters.models.join(',') : undefined,
    detections: filters.detections !== 'any' ? filters.detections : undefined,
    source: filters.source !== 'any' ? filters.source : undefined,
    since: isoParam(filters.since),
    until: isoParam(filters.until),
  }
}

function useDetections(filters) {
  return useInfiniteQuery({
    queryKey: ['detections', filterParams(filters)],
    queryFn: ({ pageParam }) =>
      apiFetch('/detections', {
        params: {
          limit: PAGE_SIZE,
          offset: pageParam,
          ...filterParams(filters),
        },
      }),
    initialPageParam: 0,
    getNextPageParam: (lastPage, allPages) =>
      lastPage.detections.length < PAGE_SIZE
        ? undefined
        : allPages.length * PAGE_SIZE,
    refetchInterval: POLL_INTERVAL_MS,
  })
}

function useFacets(filters) {
  return useQuery({
    queryKey: ['facets', filters.deviceId, filters.source],
    queryFn: () =>
      apiFetch('/detections/facets', {
        params: {
          device_id: filters.deviceId || undefined,
          source: filters.source !== 'any' ? filters.source : undefined,
        },
      }),
    refetchInterval: POLL_INTERVAL_MS,
  })
}

function useDevices() {
  return useQuery({
    queryKey: ['devices'],
    queryFn: () => apiFetch('/devices'),
    refetchInterval: DEVICES_POLL_INTERVAL_MS,
  })
}

function formatTimestamp(value) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

export default function Dashboard({ onAuthError, onLock }) {
  const queryClient = useQueryClient()
  const [filters, setFilters] = useState(filtersFromUrl)
  const [selected, setSelected] = useState(null)
  const [view, setView] = useState('detections')
  const [deleting, setDeleting] = useState(false)
  const {
    data,
    error,
    isPending,
    isFetchingNextPage,
    hasNextPage,
    fetchNextPage,
  } = useDetections(filters)
  const facetsQuery = useFacets(filters)
  const devicesQuery = useDevices()

  useEffect(() => {
    syncFiltersToUrl(filters)
  }, [filters])

  useEffect(() => {
    if (
      error?.status === 401 ||
      facetsQuery.error?.status === 401 ||
      devicesQuery.error?.status === 401
    ) {
      onAuthError()
    }
  }, [error, facetsQuery.error, devicesQuery.error, onAuthError])

  const detections = useMemo(
    () => (data ? data.pages.flatMap((page) => page.detections) : []),
    [data],
  )
  const total = data?.pages[0]?.total
  const devices = devicesQuery.data?.devices ?? []
  const positioningDevices = devices.filter(isPositioning)

  const toggleModelFilter = (hash) => {
    setFilters((current) => ({
      ...current,
      models: current.models.includes(hash)
        ? current.models.filter((entry) => entry !== hash)
        : [...current.models, hash].sort(),
    }))
  }

  // Filtering by model from the devices tab lands you on the grid that
  // filter applies to.
  const filterByModelFromDevices = (hash) => {
    toggleModelFilter(hash)
    setView('detections')
  }

  const refreshAfterDelete = () => {
    queryClient.invalidateQueries({ queryKey: ['detections'] })
    queryClient.invalidateQueries({ queryKey: ['facets'] })
  }

  const handleDeleteError = (err) => {
    if (err?.status === 401) onAuthError()
    else window.alert(`Delete failed: ${err.message}`)
  }

  const deleteSelected = async () => {
    if (!window.confirm('Delete this detection and its image? This cannot be undone.')) {
      return
    }
    try {
      await deleteDetection(selected.image_id)
      setSelected(null)
      refreshAfterDelete()
    } catch (err) {
      handleDeleteError(err)
    }
  }

  const deleteFiltered = async () => {
    const message = `Delete all ${total} matching detection${
      total === 1 ? '' : 's'
    } and their images? This cannot be undone.`
    if (!window.confirm(message)) return
    setDeleting(true)
    try {
      await deleteDetections(filterParams(filters))
      refreshAfterDelete()
    } catch (err) {
      handleDeleteError(err)
    } finally {
      setDeleting(false)
    }
  }

  // The eval view knows only image ids; fetch the full record so the
  // shared detection modal can draw both models' overlays.
  const openImage = async (imageId) => {
    try {
      const { detection } = await apiFetch(`/detections/${imageId}`)
      setSelected(detection)
    } catch {
      // Modal simply doesn't open if the record vanished.
    }
  }

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>
          Vision Classifier
          {isLocalEnvironment && <span className="env-badge">local</span>}
        </h1>
        <nav className="view-tabs" aria-label="Dashboard view">
          {[
            ['detections', 'Detections'],
            ['eval', 'Eval'],
            ['receipts', 'Receipts'],
          ].map(([value, label]) => (
            <button
              key={value}
              type="button"
              className={`view-tab${view === value ? ' active' : ''}`}
              onClick={() => setView(value)}
            >
              {label}
            </button>
          ))}
        </nav>
        <div className="header-actions">
          {/* Devices is a maintenance surface, so it sits apart from the
              primary tabs. */}
          <button
            type="button"
            className={`view-tab secondary${view === 'devices' ? ' active' : ''}`}
            onClick={() => setView('devices')}
          >
            Devices
          </button>
          <button type="button" className="ghost" onClick={onLock}>
            Lock
          </button>
        </div>
      </header>

      {positioningDevices.length > 0 && (
        <div className="positioning-banner">
          Positioning mode:{' '}
          {positioningDevices.map((device) => device.device_id).join(', ')} —
          automated capture is paused
        </div>
      )}

      {view === 'devices' && (
        <>
          <DevicesPanel
            devices={devices}
            onModelFilter={filterByModelFromDevices}
          />
          <DeviceManager
            devices={devices}
            isPending={devicesQuery.isPending}
          />
        </>
      )}

      {view === 'eval' && <EvalView onSelectImage={openImage} />}

      {view === 'receipts' && <ReceiptsPanel devices={devices} />}

      <div
        className="dashboard-view"
        style={view === 'detections' ? undefined : { display: 'none' }}
      >
      {/* Presence at a glance without the full device cards; the Devices
          tab has the previews and controls. */}
      {devices.length > 0 && (
        <div className="device-status-strip">
          {devices.map((device) => {
            const presence = presenceOf(device)
            return (
              <button
                key={device.device_id}
                type="button"
                className="device-status-chip"
                onClick={() => setView('devices')}
                title="Open the Devices tab"
              >
                <span className={`presence-dot ${presence.dot}`} />
                <span className="device-status-name">{device.device_id}</span>
                <span className="device-status-age">{presence.label}</span>
              </button>
            )
          })}
        </div>
      )}

      <div className="dashboard-body">
        <FilterSidebar
          filters={filters}
          facets={facetsQuery.data}
          onChange={setFilters}
          total={total}
          exportUrl={exportDownloadUrl(filterParams(filters))}
          onDeleteFiltered={deleteFiltered}
          deleting={deleting}
        />

        <main className="dashboard-main">
          {total !== undefined && (
            <p className="results-bar">
              {total} {total === 1 ? 'result' : 'results'}
            </p>
          )}

          {isPending && <p className="dashboard-status">Loading detections…</p>}
          {error && error.status !== 401 && (
            <p className="dashboard-status error">
              Failed to load detections: {error.message}
            </p>
          )}
          {!isPending && !error && detections.length === 0 && (
            <p className="dashboard-status">
              No detections match these filters.
            </p>
          )}

          <div className="detection-grid">
            {detections.map((detection) => (
              <DetectionCard
                key={detection.image_id}
                detection={detection}
                onClick={() => setSelected(detection)}
              />
            ))}
          </div>

          {hasNextPage && (
            <button
              type="button"
              className="load-more"
              onClick={() => fetchNextPage()}
              disabled={isFetchingNextPage}
            >
              {isFetchingNextPage ? 'Loading…' : 'Load more'}
            </button>
          )}
        </main>
      </div>
      </div>

      {selected && (
        <DetectionModal
          key={selected.image_id}
          detection={selected}
          onClose={() => setSelected(null)}
          onDelete={deleteSelected}
        />
      )}
    </div>
  )
}

function DetectionCard({ detection, onClick }) {
  const [broken, setBroken] = useState(false)
  const stored = detection.upload_status === 'stored'

  const metadata = detection.metadata ?? {}
  const boxes = overlayBoxes(metadata)
  const frameWidth = Number(metadata.frame_width)
  const frameHeight = Number(metadata.frame_height)
  const canOverlay = frameWidth > 0 && frameHeight > 0
  const counts = OVERLAY_SOURCES.filter((s) => boxes[s.key].length > 0)
  const labels = [
    ...new Set(
      OVERLAY_SOURCES.flatMap((s) => boxes[s.key].map((d) => d.label)).filter(
        Boolean,
      ),
    ),
  ]

  return (
    <button type="button" className="detection-card" onClick={onClick}>
      {stored && !broken ? (
        <div
          className="thumb-wrap"
          style={
            canOverlay
              ? { aspectRatio: `${frameWidth} / ${frameHeight}` }
              : undefined
          }
        >
          <img
            src={detectionImageUrl(detection.image_id)}
            alt={`Detection from ${detection.device_id ?? 'unknown device'}`}
            loading="lazy"
            onError={() => setBroken(true)}
          />
          {canOverlay && (
            <svg
              className="thumb-overlay"
              viewBox={`0 0 ${frameWidth} ${frameHeight}`}
              preserveAspectRatio="none"
            >
              {OVERLAY_SOURCES.map((source) =>
                boxes[source.key].map((det, index) => (
                  <BoxMarker
                    key={`${source.key}-${index}`}
                    det={det}
                    color={source.color}
                    frameWidth={frameWidth}
                    showLabel={false}
                  />
                )),
              )}
            </svg>
          )}
        </div>
      ) : (
        <div className="detection-placeholder">
          {stored ? 'image unavailable' : detection.upload_status}
        </div>
      )}
      <div className="detection-meta">
        <span className="detection-device">
          {detection.device_id ?? 'unknown device'}
        </span>
        <span className="detection-time">
          {formatTimestamp(detection.captured_at ?? detection.created_at)}
        </span>
        {(counts.length > 0 || labels.length > 0) && (
          <div className="detection-badges">
            {counts.map((source) => (
              <span
                key={source.key}
                className="badge"
                style={{ '--chip-color': source.color }}
              >
                {boxes[source.key].length} {source.label}
              </span>
            ))}
            {labels.length > 0 && (
              <span className="detection-labels">{labels.join(', ')}</span>
            )}
          </div>
        )}
      </div>
    </button>
  )
}

const OVERLAY_SOURCES = [
  { key: 'fomo', metadataKey: 'fomo_detections', label: 'FOMO', color: '#f2a65a' },
  { key: 'yolo', metadataKey: 'yolo_detections', label: 'YOLO', color: '#6fd08c' },
]

// Outline color for the ROI rectangles inference ran on (metadata
// inference_rois) — distinct from both detection sources.
const ROI_COLOR = '#b48ef2'

// Model identity fields per source: the Nicla's FOMO fields predate the
// two-model schema, so they are unprefixed in metadata.
const MODEL_STAMP_FIELDS = [
  { label: 'FOMO', hashKey: 'model_hash', manifestKey: 'model_manifest' },
  { label: 'YOLO', hashKey: 'yolo_model_hash', manifestKey: 'yolo_model_manifest' },
]

function modelStamps(metadata) {
  return MODEL_STAMP_FIELDS.flatMap((field) => {
    const hash = metadata?.[field.hashKey]
    if (!hash) return []
    const version = metadata?.[field.manifestKey]?.model_version
    return [{ label: field.label, hash, version }]
  })
}

function overlayBoxes(metadata) {
  const boxes = {}
  for (const source of OVERLAY_SOURCES) {
    const entries = metadata?.[source.metadataKey]
    boxes[source.key] = Array.isArray(entries)
      ? entries.filter((d) => Array.isArray(d.bbox) && d.bbox.length === 4)
      : []
  }
  return boxes
}

function BoxMarker({ det, color, frameWidth, showLabel = true }) {
  const [x, y, w, h] = det.bbox
  const fontSize = Math.max(9, frameWidth * 0.035)
  const label = `${det.label ?? 'object'} ${(det.confidence ?? 0).toFixed(2)}`
  const labelAbove = y > fontSize + 4
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        fill="none"
        stroke={color}
        strokeWidth={showLabel ? 2 : 1.5}
        vectorEffect="non-scaling-stroke"
      />
      {Array.isArray(det.center) && det.center.length === 2 && (
        <circle cx={det.center[0]} cy={det.center[1]} r={frameWidth * 0.008} fill={color} />
      )}
      {showLabel && (
        <text
          x={x}
          y={labelAbove ? y - 3 : y + h + fontSize}
          fill={color}
          fontSize={fontSize}
          fontFamily="ui-monospace, monospace"
          paintOrder="stroke"
          stroke="rgba(0,0,0,0.75)"
          strokeWidth={fontSize * 0.18}
        >
          {label}
        </text>
      )}
    </g>
  )
}

// Dashed outline of one inference ROI, labeled with the same index the tile
// simulator and the per-detection `tile` field use.
function RoiMarker({ roi, index, frameWidth, unit }) {
  const fontSize = Math.max(9, frameWidth * 0.03)
  return (
    <g>
      <rect
        x={roi.x}
        y={roi.y}
        width={roi.width}
        height={roi.height}
        fill="none"
        stroke={ROI_COLOR}
        strokeWidth="1.5"
        strokeDasharray="6 4"
        vectorEffect="non-scaling-stroke"
      />
      <text
        x={roi.x + 3}
        y={roi.y + fontSize}
        fill={ROI_COLOR}
        fontSize={fontSize}
        fontFamily="ui-monospace, monospace"
        paintOrder="stroke"
        stroke="rgba(0,0,0,0.75)"
        strokeWidth={fontSize * 0.18}
      >
        {unit} {index}
      </text>
    </g>
  )
}

function DetectionModal({ detection, onClose, onDelete }) {
  const metadata = detection.metadata ?? {}
  const isMotionCrops = metadata.inference_mode === 'motion_crops'

  // ROI outlines start visible for motion-crop uploads, where "what did the
  // model actually see" is the interesting question; for full sweeps the 2x3
  // grid is mostly noise, so it starts hidden but stays toggleable.
  const [visible, setVisible] = useState({
    fomo: true,
    yolo: true,
    rois: isMotionCrops,
  })

  useEffect(() => {
    function handleKey(event) {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [onClose])

  const boxes = overlayBoxes(metadata)
  const frameWidth = Number(metadata.frame_width)
  const frameHeight = Number(metadata.frame_height)
  const stored = detection.upload_status === 'stored'
  const canOverlay = stored && frameWidth > 0 && frameHeight > 0
  const hasAnyBoxes = OVERLAY_SOURCES.some((s) => boxes[s.key].length > 0)
  const rois = canOverlay ? inferenceRois(metadata) : []

  return (
    <div
      className="modal-backdrop"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <div className="modal" onClick={(event) => event.stopPropagation()}>
        <div className="modal-header">
          <h2>{detection.device_id ?? 'unknown device'}</h2>
          <div className="modal-actions">
            <button type="button" className="ghost danger" onClick={onDelete}>
              Delete
            </button>
            <button type="button" className="ghost" onClick={onClose}>
              Close
            </button>
          </div>
        </div>
        {canOverlay && (hasAnyBoxes || rois.length > 0) && (
          <div className="overlay-toggles">
            {OVERLAY_SOURCES.map((source) => (
              <button
                key={source.key}
                type="button"
                className={`chip${visible[source.key] ? ' active' : ''}`}
                style={{ '--chip-color': source.color }}
                onClick={() =>
                  setVisible((v) => ({ ...v, [source.key]: !v[source.key] }))
                }
              >
                {source.label} ({boxes[source.key].length})
              </button>
            ))}
            {rois.length > 0 && (
              <button
                type="button"
                className={`chip${visible.rois ? ' active' : ''}`}
                style={{ '--chip-color': ROI_COLOR }}
                onClick={() => setVisible((v) => ({ ...v, rois: !v.rois }))}
              >
                {isMotionCrops ? 'Motion crops' : 'Sweep tiles'} ({rois.length})
              </button>
            )}
          </div>
        )}
        {stored && (
          <div className="modal-image-wrap">
            <img
              src={detectionImageUrl(detection.image_id)}
              alt={`Detection ${detection.image_id}`}
            />
            {canOverlay && (
              <svg
                className="modal-overlay"
                viewBox={`0 0 ${frameWidth} ${frameHeight}`}
                preserveAspectRatio="none"
              >
                {visible.rois &&
                  rois.map((roi, index) => (
                    <RoiMarker
                      key={`roi-${index}`}
                      roi={roi}
                      index={index}
                      frameWidth={frameWidth}
                      unit={isMotionCrops ? 'crop' : 'tile'}
                    />
                  ))}
                {OVERLAY_SOURCES.filter((s) => visible[s.key]).map((source) =>
                  boxes[source.key].map((det, index) => (
                    <BoxMarker
                      key={`${source.key}-${index}`}
                      det={det}
                      color={source.color}
                      frameWidth={frameWidth}
                    />
                  )),
                )}
              </svg>
            )}
          </div>
        )}
        {stored && (
          <TileSimulator imageId={detection.image_id} metadata={metadata} />
        )}
        <dl className="modal-fields">
          <dt>Captured</dt>
          <dd>{formatTimestamp(detection.captured_at)}</dd>
          <dt>Received</dt>
          <dd>{formatTimestamp(detection.created_at)}</dd>
          <dt>Status</dt>
          <dd>{detection.upload_status}</dd>
          <dt>Size</dt>
          <dd>{(detection.file_size_bytes / 1024).toFixed(1)} KB</dd>
          {frameWidth > 0 && frameHeight > 0 && (
            <Fragment>
              <dt>Resolution</dt>
              <dd>
                {frameWidth}×{frameHeight}
              </dd>
            </Fragment>
          )}
          <dt>Image ID</dt>
          <dd className="mono">{detection.image_id}</dd>
          {modelStamps(metadata).map((stamp) => (
            <Fragment key={stamp.label}>
              <dt>{stamp.label} model</dt>
              <dd>
                {stamp.version ? `${stamp.version} ` : ''}
                <span className="mono">{stamp.hash}</span>
              </dd>
            </Fragment>
          ))}
        </dl>
        <details>
          <summary>Metadata</summary>
          <pre>{JSON.stringify(detection.metadata, null, 2)}</pre>
        </details>
      </div>
    </div>
  )
}
