import { useEffect, useRef, useState } from 'react'
import {
  DEFAULT_FILTERS,
  isDefaultFilters,
  isRestrictiveFilters,
} from './filterState.js'
import Sparkline from './Sparkline.jsx'

function RadioGroup({ name, value, options, onChange }) {
  return (
    <div className="filter-options">
      {options.map((option) => (
        <label key={option.value} className="filter-option">
          <input
            type="radio"
            name={name}
            checked={value === option.value}
            onChange={() => onChange(option.value)}
          />
          <span>{option.label}</span>
          {option.count !== undefined && (
            <span className="filter-count">{option.count}</span>
          )}
        </label>
      ))}
    </div>
  )
}

// Sections start collapsed so the sidebar stays short; a section with an
// active filter starts open (and pops open if a filter lands in it from
// elsewhere, e.g. the Devices tab), so a collapsed header can never hide
// what is narrowing the grid — at minimum the badge says so.
function FilterGroup({ title, badge, children }) {
  const [open, setOpen] = useState(() => Boolean(badge))
  const prevBadge = useRef(badge)

  useEffect(() => {
    if (badge && !prevBadge.current) setOpen(true)
    prevBadge.current = badge
  }, [badge])

  return (
    <div className="filter-group">
      <button
        type="button"
        className="filter-group-toggle"
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
      >
        <span>{title}</span>
        {!open && badge && <span className="filter-badge">{badge}</span>}
        <span className="filter-chevron" aria-hidden="true">
          ▸
        </span>
      </button>
      {open && <div className="filter-group-body">{children}</div>}
    </div>
  )
}

const EXPORT_MAX_IMAGES = 1000

export default function FilterSidebar({
  filters,
  facets,
  onChange,
  total,
  exportUrl,
  onDeleteFiltered,
  deleting,
}) {
  const set = (patch) => onChange({ ...filters, ...patch })

  // Keep selected labels visible even when the current facets don't include
  // them (e.g. the device filter changed), so they can be unchecked.
  const facetLabels = facets?.labels ?? []
  const knownLabels = new Set(facetLabels.map((entry) => entry.label))
  const orphanLabels = filters.labels.filter((label) => !knownLabels.has(label))
  const labelEntries = [
    ...facetLabels,
    ...orphanLabels.map((label) => ({ label, count: 0 })),
  ]

  const toggleLabel = (label) => {
    const labels = filters.labels.includes(label)
      ? filters.labels.filter((entry) => entry !== label)
      : [...filters.labels, label].sort()
    set({ labels })
  }

  // Same orphan treatment as labels: keep selected model hashes visible even
  // when the current facets no longer include them.
  const facetModels = facets?.models ?? []
  const knownModels = new Set(facetModels.map((entry) => entry.hash))
  const orphanModels = filters.models.filter((hash) => !knownModels.has(hash))
  const modelEntries = [
    ...facetModels,
    ...orphanModels.map((hash) => ({ source: '', hash, version: '', count: 0 })),
  ]

  const toggleModel = (hash) => {
    const models = filters.models.includes(hash)
      ? filters.models.filter((entry) => entry !== hash)
      : [...filters.models, hash].sort()
    set({ models })
  }

  return (
    <aside className="filters">
      <Sparkline timeline={facets?.timeline} />

      <FilterGroup
        title="Detections"
        badge={
          filters.detections !== 'any'
            ? filters.detections === 'some'
              ? 'has'
              : 'none'
            : null
        }
      >
        <RadioGroup
          name="detections"
          value={filters.detections}
          onChange={(detections) => set({ detections })}
          options={[
            { value: 'any', label: 'All', count: facets?.total },
            { value: 'some', label: 'Has detections' },
            { value: 'none', label: 'None', count: facets?.none },
          ]}
        />
      </FilterGroup>

      <FilterGroup
        title="Model"
        badge={filters.source !== 'any' ? filters.source.toUpperCase() : null}
      >
        <RadioGroup
          name="source"
          value={filters.source}
          onChange={(source) => set({ source })}
          options={[
            { value: 'any', label: 'Either' },
            { value: 'fomo', label: 'FOMO' },
            { value: 'yolo', label: 'YOLO' },
          ]}
        />
      </FilterGroup>

      <FilterGroup
        title="Model version"
        badge={filters.models.length > 0 ? String(filters.models.length) : null}
      >
        {modelEntries.length === 0 && (
          <p className="filter-empty">No stamped models yet</p>
        )}
        <div className="filter-options">
          {modelEntries.map((entry) => (
            <label
              key={`${entry.source}-${entry.hash}`}
              className="filter-option"
              title={entry.hash}
            >
              <input
                type="checkbox"
                checked={filters.models.includes(entry.hash)}
                onChange={() => toggleModel(entry.hash)}
              />
              <span>
                {entry.source ? `${entry.source.toUpperCase()} · ` : ''}
                {entry.version || entry.hash}
              </span>
              <span className="filter-count">{entry.count}</span>
            </label>
          ))}
        </div>
      </FilterGroup>

      <FilterGroup
        title="Labels"
        badge={filters.labels.length > 0 ? String(filters.labels.length) : null}
      >
        {labelEntries.length === 0 && (
          <p className="filter-empty">No labels yet</p>
        )}
        <div className="filter-options">
          {labelEntries.map((entry) => (
            <label key={entry.label} className="filter-option">
              <input
                type="checkbox"
                checked={filters.labels.includes(entry.label)}
                onChange={() => toggleLabel(entry.label)}
              />
              <span>{entry.label}</span>
              <span className="filter-count">{entry.count}</span>
            </label>
          ))}
        </div>
      </FilterGroup>

      <FilterGroup title="Device" badge={filters.deviceId || null}>
        <select
          value={filters.deviceId}
          onChange={(event) => set({ deviceId: event.target.value })}
          aria-label="Filter by device"
        >
          <option value="">All devices</option>
          {(facets?.devices ?? [])
            .filter((entry) => entry.device_id)
            .map((entry) => (
              <option key={entry.device_id} value={entry.device_id}>
                {entry.device_id} ({entry.count})
              </option>
            ))}
        </select>
      </FilterGroup>

      <FilterGroup
        title="Time range"
        badge={filters.since || filters.until ? 'set' : null}
      >
        <label className="filter-datetime">
          <span>From</span>
          <input
            type="datetime-local"
            value={filters.since}
            onChange={(event) => set({ since: event.target.value })}
          />
        </label>
        <label className="filter-datetime">
          <span>To</span>
          <input
            type="datetime-local"
            value={filters.until}
            onChange={(event) => set({ until: event.target.value })}
          />
        </label>
      </FilterGroup>

      {!isDefaultFilters(filters) && (
        <button
          type="button"
          className="ghost"
          onClick={() => onChange({ ...DEFAULT_FILTERS })}
        >
          Clear filters
        </button>
      )}

      <FilterGroup title="Export">
        {total !== undefined && total > EXPORT_MAX_IMAGES ? (
          <p className="filter-empty">
            {total} results — narrow the filters to at most {EXPORT_MAX_IMAGES}{' '}
            to export.
          </p>
        ) : (
          <a
            className={`download-button${total ? '' : ' disabled'}`}
            href={total ? exportUrl : undefined}
            aria-disabled={!total}
            onClick={(event) => {
              const message = `Download ${total} image${
                total === 1 ? '' : 's'
              } with COCO annotations as a ZIP?`
              if (!window.confirm(message)) event.preventDefault()
            }}
          >
            Download{total !== undefined ? ` (${total})` : ''}
          </a>
        )}
        <p className="filter-hint">Images + FOMO/YOLO COCO annotations</p>
      </FilterGroup>

      <FilterGroup title="Delete">
        {isRestrictiveFilters(filters) ? (
          <>
            <button
              type="button"
              className="delete-button"
              disabled={!total || deleting}
              onClick={onDeleteFiltered}
            >
              {deleting
                ? 'Deleting…'
                : `Delete${total !== undefined ? ` (${total})` : ''}`}
            </button>
            <p className="filter-hint">
              Removes every matching detection and its image
            </p>
          </>
        ) : (
          <p className="filter-hint">
            Narrow the results first — e.g. pick a time range — to delete
            what matches.
          </p>
        )}
      </FilterGroup>
    </aside>
  )
}
