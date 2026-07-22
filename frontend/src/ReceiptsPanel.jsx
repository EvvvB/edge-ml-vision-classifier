import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { apiFetch } from './api.js'

// The Pi ships receipts every minute, so poll on the same cadence as the
// detection grid.
const RECEIPTS_POLL_INTERVAL_MS = 30_000
const RECEIPTS_LIMIT = 200

const EVENT_STYLES = {
  accepted: { color: '#6fd08c' },
  rejected: { color: '#e66a6a' },
}

function formatTimestamp(value) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

// The Pi's arrival log: one row per upload a camera sent, including the
// rejected ones no other tab can show (rejected frames are never stored).
export default function ReceiptsPanel({ devices }) {
  const [event, setEvent] = useState('any')
  const [deviceId, setDeviceId] = useState('')

  const receiptsQuery = useQuery({
    queryKey: ['receipts', event, deviceId],
    queryFn: () =>
      apiFetch('/receipts', {
        params: {
          event: event !== 'any' ? event : undefined,
          device_id: deviceId || undefined,
          limit: RECEIPTS_LIMIT,
        },
      }),
    refetchInterval: RECEIPTS_POLL_INTERVAL_MS,
  })

  const receipts = receiptsQuery.data?.receipts ?? []
  // Receipts can name devices the registry has pruned; the filter should
  // still be able to select them.
  const deviceOptions = [
    ...new Set([
      ...devices.map((device) => device.device_id),
      ...receipts.map((receipt) => receipt.device_id).filter(Boolean),
    ]),
  ].sort()

  return (
    <div className="receipts-view">
      <div className="receipts-header">
        <h2>Upload receipts</h2>
        <span className="eval-panel-note">
          every upload the Pi received from a camera — rejections included.
          Synced from the Pi every minute.
        </span>
        <div className="receipt-filters">
          {['any', 'accepted', 'rejected'].map((value) => (
            <button
              key={value}
              type="button"
              className={`chip${event === value ? ' active' : ''}`}
              style={
                value !== 'any'
                  ? { '--chip-color': EVENT_STYLES[value].color }
                  : { '--chip-color': 'var(--accent)' }
              }
              onClick={() => setEvent(value)}
            >
              {value === 'any' ? 'All' : value}
            </button>
          ))}
          <select
            value={deviceId}
            onChange={(changeEvent) => setDeviceId(changeEvent.target.value)}
            aria-label="Filter by device"
          >
            <option value="">all devices</option>
            {deviceOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </div>
      </div>

      {receiptsQuery.isPending && (
        <p className="dashboard-status">Loading receipts…</p>
      )}
      {receiptsQuery.error && receiptsQuery.error.status !== 401 && (
        <p className="dashboard-status error">
          Failed to load receipts: {receiptsQuery.error.message}
        </p>
      )}
      {receiptsQuery.data && receipts.length === 0 && (
        <p className="dashboard-status">
          No receipts yet — they appear within a minute of the next upload.
        </p>
      )}

      {receipts.length > 0 && (
        <div className="device-table-wrap">
          <table className="device-table">
            <thead>
              <tr>
                <th>Received</th>
                <th>Event</th>
                <th>Device</th>
                <th>File</th>
                <th>FOMO</th>
                <th>Detail</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {receipts.map((receipt) => (
                <tr key={receipt.receipt_id}>
                  <td>{formatTimestamp(receipt.logged_at ?? receipt.created_at)}</td>
                  <td>
                    <span
                      className="badge"
                      style={{
                        '--chip-color':
                          EVENT_STYLES[receipt.event]?.color ?? 'var(--text-dim)',
                      }}
                    >
                      {receipt.event}
                    </span>
                  </td>
                  <td>{receipt.device_id ?? '—'}</td>
                  <td className="mono">{receipt.filename ?? '—'}</td>
                  <td>{receipt.fomo_count ?? '—'}</td>
                  <td className="receipt-detail">
                    {receipt.event === 'rejected'
                      ? receipt.reason ?? '—'
                      : receipt.image_id ?? '—'}
                  </td>
                  <td className="receipt-source">
                    {[receipt.pi_id, receipt.client_host]
                      .filter(Boolean)
                      .join(' · ') || '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {receipts.length === RECEIPTS_LIMIT && (
        <p className="eval-panel-note">
          Showing the latest {RECEIPTS_LIMIT} receipts.
        </p>
      )}
    </div>
  )
}
