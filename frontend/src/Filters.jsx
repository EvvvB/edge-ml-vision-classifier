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

const EXPORT_MAX_IMAGES = 1000

export default function FilterSidebar({
  filters,
  facets,
  onChange,
  total,
  exportUrl,
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
        <h3>Model version</h3>
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

      <div className="filter-group">
        <h3>Export</h3>
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
      </div>
    </aside>
  )
}
