import { useEffect, useMemo, useState } from 'react'
import { useInfiniteQuery, useQuery } from '@tanstack/react-query'
import { apiFetch, detectionImageUrl, exportDownloadUrl } from './api.js'
import FilterSidebar from './Filters.jsx'
import { filtersFromUrl, syncFiltersToUrl } from './filterState.js'

const PAGE_SIZE = 24
const POLL_INTERVAL_MS = 30_000

function filterParams(filters) {
  return {
    device_id: filters.deviceId || undefined,
    labels: filters.labels.length > 0 ? filters.labels.join(',') : undefined,
    detections: filters.detections !== 'any' ? filters.detections : undefined,
    source: filters.source !== 'any' ? filters.source : undefined,
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

function formatTimestamp(value) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

export default function Dashboard({ onAuthError, onLock }) {
  const [filters, setFilters] = useState(filtersFromUrl)
  const [selected, setSelected] = useState(null)
  const {
    data,
    error,
    isPending,
    isFetchingNextPage,
    hasNextPage,
    fetchNextPage,
  } = useDetections(filters)
  const facetsQuery = useFacets(filters)

  useEffect(() => {
    syncFiltersToUrl(filters)
  }, [filters])

  useEffect(() => {
    if (error?.status === 401 || facetsQuery.error?.status === 401) {
      onAuthError()
    }
  }, [error, facetsQuery.error, onAuthError])

  const detections = useMemo(
    () => (data ? data.pages.flatMap((page) => page.detections) : []),
    [data],
  )
  const total = data?.pages[0]?.total

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>Vision Classifier</h1>
        <button type="button" className="ghost" onClick={onLock}>
          Lock
        </button>
      </header>

      <div className="dashboard-body">
        <FilterSidebar
          filters={filters}
          facets={facetsQuery.data}
          onChange={setFilters}
          total={total}
          exportUrl={exportDownloadUrl(filterParams(filters))}
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

      {selected && (
        <DetectionModal
          detection={selected}
          onClose={() => setSelected(null)}
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

function DetectionModal({ detection, onClose }) {
  const [visible, setVisible] = useState({ fomo: true, yolo: true })

  useEffect(() => {
    function handleKey(event) {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [onClose])

  const metadata = detection.metadata ?? {}
  const boxes = overlayBoxes(metadata)
  const frameWidth = Number(metadata.frame_width)
  const frameHeight = Number(metadata.frame_height)
  const stored = detection.upload_status === 'stored'
  const canOverlay = stored && frameWidth > 0 && frameHeight > 0
  const hasAnyBoxes = OVERLAY_SOURCES.some((s) => boxes[s.key].length > 0)

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
          <button type="button" className="ghost" onClick={onClose}>
            Close
          </button>
        </div>
        {canOverlay && hasAnyBoxes && (
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
        <dl className="modal-fields">
          <dt>Captured</dt>
          <dd>{formatTimestamp(detection.captured_at)}</dd>
          <dt>Received</dt>
          <dd>{formatTimestamp(detection.created_at)}</dd>
          <dt>Status</dt>
          <dd>{detection.upload_status}</dd>
          <dt>Size</dt>
          <dd>{(detection.file_size_bytes / 1024).toFixed(1)} KB</dd>
          <dt>Image ID</dt>
          <dd className="mono">{detection.image_id}</dd>
        </dl>
        <details>
          <summary>Metadata</summary>
          <pre>{JSON.stringify(detection.metadata, null, 2)}</pre>
        </details>
      </div>
    </div>
  )
}
