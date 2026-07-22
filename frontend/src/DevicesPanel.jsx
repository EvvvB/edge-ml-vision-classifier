import { useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import {
  previewImageUrl,
  requestCapture,
  setDeviceConfig,
  setDeviceMode,
} from './api.js'

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

// Values the camera firmware accepts; anything else it would ignore.
const SWEEP_INTERVAL_OPTIONS = [
  { label: '5 s', value: 5_000 },
  { label: '30 s', value: 30_000 },
  { label: '5 min', value: 300_000 },
  { label: '10 min', value: 600_000 },
  { label: '30 min', value: 1_800_000 },
  { label: '1 h', value: 3_600_000 },
]

const CROP_SIZE_OPTIONS = [
  { label: '96 px — sharp', value: 96 },
  { label: '192 px — wide', value: 192 },
]

// Per-pixel intensity delta (0-255) that counts as motion; lower is more
// sensitive. Labels lead with sensitivity so the direction is obvious.
const MOTION_THRESHOLD_OPTIONS = [
  { label: 'Very sensitive (8)', value: 8 },
  { label: 'Sensitive (16)', value: 16 },
  { label: 'Default (24)', value: 24 },
  { label: 'Tolerant (40)', value: 40 },
  { label: 'Very tolerant (64)', value: 64 },
]

// 'off' maps to model_enabled: false on the wire — a separate knob from
// min_confidence, so "run the model, keep everything" (0) stays sayable.
const MODEL_OPTIONS = [
  { label: 'Off — motion only', value: 'off' },
  { label: '0 — keep all', value: 0 },
  { label: '0.35', value: 0.35 },
  { label: '0.5', value: 0.5 },
  { label: '0.7', value: 0.7 },
]

function configIsPending(device) {
  return (
    (device.desired_config_seq ?? 0) > 0 &&
    (device.reported_config_seq ?? -1) < device.desired_config_seq
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

      <ConfigControls device={device} disabled={offline} />

      {positioning && !offline && <PreviewImage deviceId={device.device_id} />}
    </div>
  )
}

function ConfigControls({ device, disabled }) {
  const queryClient = useQueryClient()
  const [saving, setSaving] = useState(false)
  const timeoutsRef = useRef([])

  useEffect(() => () => timeoutsRef.current.forEach(clearTimeout), [])

  const desired = device.desired_config ?? {}
  const reported = device.reported_config
  const pending = configIsPending(device)

  async function updateConfig(patch) {
    setSaving(true)
    try {
      await setDeviceConfig(device.device_id, patch)
      // Same round trip as a mode switch: device -> Pi -> cloud in a few
      // seconds, so refresh a couple of times to resolve "waiting".
      for (const delay of [0, 2500, 6000]) {
        timeoutsRef.current.push(
          setTimeout(
            () => queryClient.invalidateQueries({ queryKey: ['devices'] }),
            delay,
          ),
        )
      }
    } finally {
      setSaving(false)
    }
  }

  return (
    <div
      className="device-card-row"
      title={
        reported
          ? `Camera reports: sweep ${
              (reported.full_sweep_interval_ms ?? 0) / 1000
            } s · crop ${reported.crop_size ?? '?'} px · motion ${
              reported.motion_diff_threshold ?? '?'
            } · ${
              reported.model_enabled === false
                ? 'model off'
                : `confidence ${reported.min_confidence ?? '?'}`
            }`
          : 'The camera has not reported its config yet'
      }
    >
      <label className="config-field">
        Sweep
        <select
          value={desired.full_sweep_interval_ms ?? ''}
          disabled={saving || disabled}
          onChange={(event) =>
            updateConfig({ full_sweep_interval_ms: Number(event.target.value) })
          }
        >
          {desired.full_sweep_interval_ms == null && (
            <option value="">firmware default</option>
          )}
          {SWEEP_INTERVAL_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </label>

      <label className="config-field">
        Crop
        <select
          value={desired.crop_size ?? ''}
          disabled={saving || disabled}
          onChange={(event) =>
            updateConfig({ crop_size: Number(event.target.value) })
          }
        >
          {desired.crop_size == null && (
            <option value="">firmware default</option>
          )}
          {CROP_SIZE_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </label>

      <label className="config-field">
        Motion
        <select
          value={desired.motion_diff_threshold ?? ''}
          disabled={saving || disabled}
          onChange={(event) =>
            updateConfig({ motion_diff_threshold: Number(event.target.value) })
          }
        >
          {desired.motion_diff_threshold == null && (
            <option value="">firmware default</option>
          )}
          {MOTION_THRESHOLD_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </label>

      <label className="config-field">
        Model
        <select
          value={
            desired.model_enabled === false
              ? 'off'
              : desired.min_confidence ?? ''
          }
          disabled={saving || disabled}
          onChange={(event) =>
            updateConfig(
              event.target.value === 'off'
                ? { model_enabled: false }
                : {
                    model_enabled: true,
                    min_confidence: Number(event.target.value),
                  },
            )
          }
        >
          {desired.model_enabled !== false &&
            desired.min_confidence == null && (
              <option value="">firmware default</option>
            )}
          {MODEL_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </label>

      {pending && !disabled && (
        <span className="mode-pending">waiting for device…</span>
      )}
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
