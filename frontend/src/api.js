const KEY_STORAGE = 'cloud-api-key'

export function getStoredKey() {
  return localStorage.getItem(KEY_STORAGE) || ''
}

export function storeKey(key) {
  localStorage.setItem(KEY_STORAGE, key)
}

export function clearKey() {
  localStorage.removeItem(KEY_STORAGE)
}

export class ApiError extends Error {
  constructor(status, message) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export async function apiFetch(path, { params, method, body } = {}) {
  const url = new URL(path, window.location.origin)
  for (const [name, value] of Object.entries(params ?? {})) {
    if (value !== undefined && value !== null && value !== '') {
      url.searchParams.set(name, value)
    }
  }

  const key = getStoredKey()
  const headers = key ? { 'X-API-Key': key } : {}
  const init = { method: method ?? 'GET', headers }
  if (body !== undefined) {
    headers['Content-Type'] = 'application/json'
    init.body = JSON.stringify(body)
  }
  const response = await fetch(url, init)

  if (!response.ok) {
    let detail = response.statusText
    try {
      detail = (await response.json()).detail ?? detail
    } catch {
      // body was not JSON; keep the status text
    }
    throw new ApiError(response.status, detail)
  }
  return response.json()
}

export function requestCapture(deviceId) {
  return apiFetch(`/devices/${encodeURIComponent(deviceId)}/capture`, {
    method: 'POST',
  })
}

export function setDeviceMode(deviceId, mode) {
  return apiFetch(`/devices/${encodeURIComponent(deviceId)}/mode`, {
    method: 'POST',
    body: { mode },
  })
}

export function deleteDevice(deviceId) {
  return apiFetch(`/devices/${encodeURIComponent(deviceId)}`, {
    method: 'DELETE',
  })
}

// Preview frames render in an <img>, so the key rides as a query parameter
// like the detection images.
export function previewImageUrl(deviceId) {
  const key = getStoredKey()
  const query = key ? `?key=${encodeURIComponent(key)}` : ''
  return `/devices/${encodeURIComponent(deviceId)}/preview${query}`
}

// <img> tags cannot send headers, so the image endpoint takes the key as a
// query parameter instead.
export function detectionImageUrl(imageId) {
  const key = getStoredKey()
  const query = key ? `?key=${encodeURIComponent(key)}` : ''
  return `/detections/${imageId}/image${query}`
}

// Downloads are browser navigations (no headers), so the key rides along as a
// query parameter here too.
export function exportDownloadUrl(params) {
  const url = new URL('/detections/export', window.location.origin)
  for (const [name, value] of Object.entries(params ?? {})) {
    if (value !== undefined && value !== null && value !== '') {
      url.searchParams.set(name, value)
    }
  }
  const key = getStoredKey()
  if (key) url.searchParams.set('key', key)
  return url.pathname + url.search
}
