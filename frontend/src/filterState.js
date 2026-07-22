export const DEFAULT_FILTERS = {
  detections: 'any',
  source: 'any',
  labels: [],
  models: [],
  deviceId: '',
  since: '',
  until: '',
}

function listFromParam(value) {
  return (value || '')
    .split(',')
    .map((entry) => entry.trim().toLowerCase())
    .filter(Boolean)
}

// since/until hold the raw datetime-local input value ("2026-07-22T14:30",
// local time); anything unparseable is treated as unset.
function datetimeFromParam(value) {
  if (!value || Number.isNaN(new Date(value).getTime())) return ''
  return value
}

export function filtersFromUrl() {
  const params = new URLSearchParams(window.location.search)
  const detections = params.get('detections')
  const source = params.get('source')
  return {
    detections: ['some', 'none'].includes(detections) ? detections : 'any',
    source: ['fomo', 'yolo'].includes(source) ? source : 'any',
    labels: listFromParam(params.get('labels')),
    models: listFromParam(params.get('models')),
    deviceId: params.get('device') || '',
    since: datetimeFromParam(params.get('since')),
    until: datetimeFromParam(params.get('until')),
  }
}

export function syncFiltersToUrl(filters) {
  const params = new URLSearchParams()
  if (filters.detections !== 'any') params.set('detections', filters.detections)
  if (filters.source !== 'any') params.set('source', filters.source)
  if (filters.labels.length > 0) params.set('labels', filters.labels.join(','))
  if (filters.models.length > 0) params.set('models', filters.models.join(','))
  if (filters.deviceId) params.set('device', filters.deviceId)
  if (filters.since) params.set('since', filters.since)
  if (filters.until) params.set('until', filters.until)
  const query = params.toString()
  window.history.replaceState(
    null,
    '',
    query ? `?${query}` : window.location.pathname,
  )
}

export function isDefaultFilters(filters) {
  return (
    filters.detections === 'any' &&
    filters.source === 'any' &&
    filters.labels.length === 0 &&
    filters.models.length === 0 &&
    !filters.deviceId &&
    !filters.since &&
    !filters.until
  )
}

// Whether the filters actually narrow the result set. A source filter alone
// does not (it only scopes what the other filters look at), and the API
// refuses a bulk delete that would match everything.
export function isRestrictiveFilters(filters) {
  return (
    filters.detections !== 'any' ||
    filters.labels.length > 0 ||
    filters.models.length > 0 ||
    Boolean(filters.deviceId) ||
    Boolean(filters.since) ||
    Boolean(filters.until)
  )
}
