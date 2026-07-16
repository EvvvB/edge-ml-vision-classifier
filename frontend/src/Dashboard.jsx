import { useEffect, useMemo, useState } from 'react'
import { useInfiniteQuery } from '@tanstack/react-query'
import { apiFetch, detectionImageUrl } from './api.js'

const PAGE_SIZE = 24
const POLL_INTERVAL_MS = 30_000

function useDetections(deviceId) {
  return useInfiniteQuery({
    queryKey: ['detections', deviceId],
    queryFn: ({ pageParam }) =>
      apiFetch('/detections', {
        params: {
          limit: PAGE_SIZE,
          offset: pageParam,
          device_id: deviceId || undefined,
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

function formatTimestamp(value) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

export default function Dashboard({ onAuthError, onLock }) {
  const [deviceId, setDeviceId] = useState('')
  const [selected, setSelected] = useState(null)
  const {
    data,
    error,
    isPending,
    isFetchingNextPage,
    hasNextPage,
    fetchNextPage,
  } = useDetections(deviceId)

  useEffect(() => {
    if (error?.status === 401) onAuthError()
  }, [error, onAuthError])

  const detections = useMemo(
    () => (data ? data.pages.flatMap((page) => page.detections) : []),
    [data],
  )

  const deviceIds = useMemo(() => {
    const ids = new Set(detections.map((d) => d.device_id).filter(Boolean))
    if (deviceId) ids.add(deviceId)
    return [...ids].sort()
  }, [detections, deviceId])

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>Vision Classifier</h1>
        <div className="dashboard-controls">
          <select
            value={deviceId}
            onChange={(event) => setDeviceId(event.target.value)}
            aria-label="Filter by device"
          >
            <option value="">All devices</option>
            {deviceIds.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
          <button type="button" className="ghost" onClick={onLock}>
            Lock
          </button>
        </div>
      </header>

      {isPending && <p className="dashboard-status">Loading detections…</p>}
      {error && error.status !== 401 && (
        <p className="dashboard-status error">
          Failed to load detections: {error.message}
        </p>
      )}
      {!isPending && !error && detections.length === 0 && (
        <p className="dashboard-status">No detections yet.</p>
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
  return (
    <button type="button" className="detection-card" onClick={onClick}>
      {stored && !broken ? (
        <img
          src={detectionImageUrl(detection.image_id)}
          alt={`Detection from ${detection.device_id ?? 'unknown device'}`}
          loading="lazy"
          onError={() => setBroken(true)}
        />
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
      </div>
    </button>
  )
}

function DetectionModal({ detection, onClose }) {
  useEffect(() => {
    function handleKey(event) {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [onClose])

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
        {detection.upload_status === 'stored' && (
          <img
            src={detectionImageUrl(detection.image_id)}
            alt={`Detection ${detection.image_id}`}
          />
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
