import { useState } from 'react'

const WIDTH = 200
const HEIGHT = 52
const PAD_X = 2
const PAD_TOP = 6
const PAD_BOTTOM = 3
// Validated against the dark surface (dataviz palette checks).
const SERIES_COLOR = '#3d94d9'

function hourLabel(iso) {
  return new Date(iso).toLocaleTimeString([], { hour: 'numeric' })
}

export default function Sparkline({ timeline }) {
  const [hovered, setHovered] = useState(null)

  if (!Array.isArray(timeline) || timeline.length < 2) return null

  const counts = timeline.map((entry) => entry.count)
  const total = counts.reduce((sum, count) => sum + count, 0)
  const max = Math.max(...counts, 1)
  const stepX = (WIDTH - PAD_X * 2) / (timeline.length - 1)
  const pointX = (index) => PAD_X + index * stepX
  const pointY = (count) =>
    PAD_TOP + (1 - count / max) * (HEIGHT - PAD_TOP - PAD_BOTTOM)

  const line = timeline
    .map((entry, index) => `${pointX(index)},${pointY(entry.count)}`)
    .join(' ')
  const baseline = HEIGHT - PAD_BOTTOM
  const area = `${PAD_X},${baseline} ${line} ${pointX(timeline.length - 1)},${baseline}`

  const indexFromEvent = (event) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const fraction = (event.clientX - rect.left) / rect.width
    const index = Math.round(((fraction * WIDTH) - PAD_X) / stepX)
    return Math.min(timeline.length - 1, Math.max(0, index))
  }

  const active = hovered === null ? null : timeline[hovered]

  return (
    <div className="filter-group">
      <h3>Last 24 h</h3>
      <p className="sparkline-value">
        {active
          ? `${active.count} at ${hourLabel(active.hour)}`
          : `${total} ${total === 1 ? 'detection' : 'detections'}`}
      </p>
      <svg
        className="sparkline"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`Detections per hour over the last 24 hours, ${total} total`}
        onMouseMove={(event) => setHovered(indexFromEvent(event))}
        onMouseLeave={() => setHovered(null)}
      >
        <polygon points={area} fill={SERIES_COLOR} opacity="0.16" />
        <polyline
          points={line}
          fill="none"
          stroke={SERIES_COLOR}
          strokeWidth="2"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {active && (
          <g>
            <line
              x1={pointX(hovered)}
              y1={PAD_TOP}
              x2={pointX(hovered)}
              y2={baseline}
              stroke="currentColor"
              opacity="0.25"
            />
            <circle
              cx={pointX(hovered)}
              cy={pointY(active.count)}
              r="3.5"
              fill={SERIES_COLOR}
              stroke="var(--surface)"
              strokeWidth="2"
            />
          </g>
        )}
      </svg>
    </div>
  )
}
