import { useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { previewImageUrl, requestCapture, setDeviceMode } from './api.js'

// The cloud's last_seen freshness is bounded by the Pi's ~5-minute relay
// throttle, so presence thresholds are multiples of that, not of the
// camera's own sweep cadence.
const SEEN_FRESH_MS = 12 * 60_000
const SEEN_STALE_MS = 30 * 60_000

function formatAge(ms) {
  if (ms < 90_000) return 'just now'
  const minutes = Math.round(ms / 60_000)
  if (minutes < 90) return `${minutes} min ago`
  const hours = Math.round(minutes / 60)
  if (hours < 36) return `${hours} h ago`
  return `${Math.round(hours / 24)} d ago`
}

export function presenceOf(device) {
  if (!device.gateway_connected) {
    return { dot: 'gray', label: 'gateway offline' }
  }
  const seen = device.last_seen_at ? Date.parse(device.last_seen_at) : NaN
  if (Number.isNaN(seen)) {
    return { dot: 'gray', label: 'never seen' }
  }
  const age = Date.now() - seen
  const label = `seen ${formatAge(age)}`
  if (age <= SEEN_FRESH_MS) return { dot: 'green', label }
  if (age <= SEEN_STALE_MS) return { dot: 'amber', label }
  return { dot: 'gray', label }
}

export function isPositioning(device) {
  return (
    device.desired_mode === 'positioning' ||
    device.reported_mode === 'positioning'
  )
}

function modeIsPending(device) {
  return (
    (device.desired_mode_seq ?? 0) > 0 &&
    (device.reported_mode_seq ?? -1) < device.desired_mode_seq
  )
}

export default function DevicesPanel({ devices, onModelFilter }) {
  if (!devices || devices.length === 0) return null

  return (
    <div className="devices-panel">
      {devices.map((device) => (
        <DeviceCard
          key={device.device_id}
          device={device}
          onModelFilter={onModelFilter}
        />
      ))}
    </div>
  )
}

function DeviceCard({ device, onModelFilter }) {
  const queryClient = useQueryClient()
  const [switching, setSwitching] = useState(false)
  const timeoutsRef = useRef([])

  useEffect(() => () => timeoutsRef.current.forEach(clearTimeout), [])

  const presence = presenceOf(device)
  const pending = modeIsPending(device)
  const positioning = isPositioning(device)
  const modelVersion = device.model_manifest?.model_version
  // A gray dot means commands cannot reach the device right now (gateway
  // down, or the camera itself has gone quiet). Disabling the controls
  // keeps desired state from drifting further while nothing can sync.
  const offline = presence.dot === 'gray'

  async function switchMode(mode) {
    if (mode === device.desired_mode) return
    setSwitching(true)
    try {
      await setDeviceMode(device.device_id, mode)
      // The ack round-trips device -> Pi -> cloud in a few seconds;
      // refresh a couple of times so "waiting" resolves without the poll.
      for (const delay of [0, 2500, 6000]) {
        timeoutsRef.current.push(
          setTimeout(
            () => queryClient.invalidateQueries({ queryKey: ['devices'] }),
            delay,
          ),
        )
      }
    } finally {
      setSwitching(false)
    }
  }

  return (
    <div
      className={`device-card${positioning ? ' positioning' : ''}${
        offline ? ' offline' : ''
      }`}
    >
      <div className="device-card-header">
        <span className={`presence-dot ${presence.dot}`} />
        <span className="device-name">
          {device.display_name || device.device_id}
        </span>
        {offline && <span className="offline-badge">offline</span>}
        <span className="device-seen">{presence.label}</span>
      </div>

      <div className="device-card-row">
        {device.model_hash && (
          <button
            type="button"
            className="chip"
            title={`Filter detections by model ${device.model_hash}`}
            onClick={() => onModelFilter?.(device.model_hash)}
          >
            FOMO · {modelVersion || device.model_hash}
          </button>
        )}
        {device.firmware_build && (
          <span className="device-firmware">fw {device.firmware_build}</span>
        )}
      </div>

      <div className="device-card-row">
        <div className="mode-toggle" role="group" aria-label="Camera mode">
          {['automated', 'positioning'].map((mode) => (
            <button
              key={mode}
              type="button"
              className={`chip${device.desired_mode === mode ? ' active' : ''}`}
              disabled={switching || offline}
              title={offline ? `${presence.label} — controls resume when it reconnects` : undefined}
              onClick={() => switchMode(mode)}
            >
              {mode === 'automated' ? 'Auto' : 'Positioning'}
            </button>
          ))}
        </div>
        {pending && !offline && (
          <span className="mode-pending">waiting for device…</span>
        )}
        <CaptureButton deviceId={device.device_id} disabled={offline} />
      </div>

      {positioning && !offline && <PreviewImage deviceId={device.device_id} />}
    </div>
  )
}

function PreviewImage({ deviceId }) {
  const [tick, setTick] = useState(0)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    const interval = setInterval(() => {
      setFailed(false)
      setTick((value) => value + 1)
    }, 1200)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="device-preview">
      {failed ? (
        <p className="device-preview-empty">Waiting for preview frames…</p>
      ) : (
        <img
          src={`${previewImageUrl(deviceId)}&t=${tick}`}
          alt={`Positioning preview from ${deviceId}`}
          onError={() => setFailed(true)}
        />
      )}
    </div>
  )
}

function CaptureButton({ deviceId, disabled = false }) {
  const queryClient = useQueryClient()
  const [status, setStatus] = useState('idle')
  const timeoutsRef = useRef([])

  useEffect(() => () => timeoutsRef.current.forEach(clearTimeout), [])

  const schedule = (fn, ms) => {
    timeoutsRef.current.push(setTimeout(fn, ms))
  }

  async function handleCapture() {
    setStatus('requesting')
    try {
      await requestCapture(deviceId)
      setStatus('requested')
      // The frame takes several seconds to travel Nicla -> Pi -> cloud;
      // refresh the grid a couple of times so it appears without waiting
      // for the 30s poll.
      for (const delay of [6000, 14000]) {
        schedule(() => {
          queryClient.invalidateQueries({ queryKey: ['detections'] })
          queryClient.invalidateQueries({ queryKey: ['facets'] })
        }, delay)
      }
      schedule(() => setStatus('idle'), 5000)
    } catch {
      setStatus('error')
      schedule(() => setStatus('idle'), 5000)
    }
  }

  const labels = {
    idle: 'Capture',
    requesting: 'Requesting…',
    requested: 'Requested ✓',
    error: 'Failed',
  }

  return (
    <button
      type="button"
      className={`capture-button${status === 'error' ? ' error' : ''}`}
      onClick={handleCapture}
      disabled={
        disabled ||
        !deviceId ||
        status === 'requesting' ||
        status === 'requested'
      }
      title={
        disabled
          ? 'Device is offline'
          : `Capture a frame from ${deviceId}`
      }
    >
      {labels[status]}
    </button>
  )
}
