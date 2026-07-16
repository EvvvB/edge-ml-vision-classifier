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

export async function apiFetch(path, { params } = {}) {
  const url = new URL(path, window.location.origin)
  for (const [name, value] of Object.entries(params ?? {})) {
    if (value !== undefined && value !== null && value !== '') {
      url.searchParams.set(name, value)
    }
  }

  const key = getStoredKey()
  const response = await fetch(url, {
    headers: key ? { 'X-API-Key': key } : {},
  })

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

// <img> tags cannot send headers, so the image endpoint takes the key as a
// query parameter instead.
export function detectionImageUrl(imageId) {
  const key = getStoredKey()
  const query = key ? `?key=${encodeURIComponent(key)}` : ''
  return `/detections/${imageId}/image${query}`
}
