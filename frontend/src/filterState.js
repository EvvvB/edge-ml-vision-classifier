export const DEFAULT_FILTERS = {
  detections: 'any',
  source: 'any',
  labels: [],
  models: [],
  deviceId: '',
}

function listFromParam(value) {
  return (value || '')
    .split(',')
    .map((entry) => entry.trim().toLowerCase())
    .filter(Boolean)
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
  }
}

export function syncFiltersToUrl(filters) {
  const params = new URLSearchParams()
  if (filters.detections !== 'any') params.set('detections', filters.detections)
  if (filters.source !== 'any') params.set('source', filters.source)
  if (filters.labels.length > 0) params.set('labels', filters.labels.join(','))
  if (filters.models.length > 0) params.set('models', filters.models.join(','))
  if (filters.deviceId) params.set('device', filters.deviceId)
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
    !filters.deviceId
  )
}
