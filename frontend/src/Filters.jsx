import { DEFAULT_FILTERS, isDefaultFilters } from './filterState.js'
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

export default function FilterSidebar({ filters, facets, onChange }) {
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

  return (
    <aside className="filters">
      <Sparkline timeline={facets?.timeline} />

      <div className="filter-group">
        <h3>Detections</h3>
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
      </div>

      <div className="filter-group">
        <h3>Model</h3>
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
      </div>

      <div className="filter-group">
        <h3>Labels</h3>
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
      </div>

      <div className="filter-group">
        <h3>Device</h3>
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
      </div>

      {!isDefaultFilters(filters) && (
        <button
          type="button"
          className="ghost"
          onClick={() => onChange({ ...DEFAULT_FILTERS })}
        >
          Clear filters
        </button>
      )}
    </aside>
  )
}
