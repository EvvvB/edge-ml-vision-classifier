import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { deleteDevice } from './api.js'
import { presenceOf } from './DevicesPanel.jsx'

function formatTimestamp(value) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

export default function DeviceManager({ devices, isPending }) {
  if (isPending) {
    return <p className="dashboard-status">Loading devices…</p>
  }
  if (!devices || devices.length === 0) {
    return (
      <p className="dashboard-status">
        No devices registered yet. A camera registers itself the first time
        it hellos or uploads.
      </p>
    )
  }

  return (
    <div className="device-manager">
      <p className="device-manager-hint">
        Pruning removes a device's registry row and capture counter. Its
        detections stay queryable, and the device simply re-registers if it
        ever hellos again.
      </p>
      <div className="device-table-wrap">
        <table className="device-table">
          <thead>
            <tr>
              <th>Device</th>
              <th>Hardware ID</th>
              <th>Firmware</th>
              <th>Model</th>
              <th>Gateway</th>
              <th>First seen</th>
              <th>Last hello</th>
              <th>Last upload</th>
              <th>Mode</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {devices.map((device) => (
              <DeviceRow key={device.device_id} device={device} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function DeviceRow({ device }) {
  const queryClient = useQueryClient()
  const [status, setStatus] = useState('idle')
  const presence = presenceOf(device)
  const modelVersion = device.model_manifest?.model_version

  async function handleRemove() {
    const message =
      `Remove ${device.device_id} from the registry? ` +
      'Its detections are kept.'
    if (!window.confirm(message)) return
    setStatus('removing')
    try {
      await deleteDevice(device.device_id)
      queryClient.invalidateQueries({ queryKey: ['devices'] })
    } catch {
      setStatus('error')
      setTimeout(() => setStatus('idle'), 4000)
    }
  }

  return (
    <tr>
      <td>
        <span className={`presence-dot ${presence.dot}`} />{' '}
        <span className="device-name">
          {device.display_name || device.device_id}
        </span>
        <div className="device-table-sub">{presence.label}</div>
      </td>
      <td className="mono">{device.hardware_id ?? '—'}</td>
      <td className="mono">{device.firmware_build ?? '—'}</td>
      <td>
        {device.model_hash ? (
          <>
            {modelVersion ? `${modelVersion} ` : ''}
            <span className="mono">{device.model_hash}</span>
          </>
        ) : (
          '—'
        )}
      </td>
      <td>
        {device.pi_id ?? '—'}
        <div className="device-table-sub">
          {device.gateway_connected ? 'connected' : 'offline'}
        </div>
      </td>
      <td>{formatTimestamp(device.first_seen_at)}</td>
      <td>{formatTimestamp(device.last_hello_at)}</td>
      <td>{formatTimestamp(device.last_upload_at)}</td>
      <td>
        {device.desired_mode}
        {device.reported_mode && device.reported_mode !== device.desired_mode && (
          <div className="device-table-sub">
            reports {device.reported_mode}
          </div>
        )}
      </td>
      <td>
        <button
          type="button"
          className="ghost danger"
          onClick={handleRemove}
          disabled={status === 'removing'}
        >
          {status === 'error' ? 'Failed' : 'Remove'}
        </button>
      </td>
    </tr>
  )
}
