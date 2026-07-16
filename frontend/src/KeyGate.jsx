import { useState } from 'react'
import { apiFetch, clearKey, storeKey } from './api.js'

export default function KeyGate({ onUnlocked }) {
  const [input, setInput] = useState('')
  const [error, setError] = useState('')
  const [checking, setChecking] = useState(false)

  async function handleSubmit(event) {
    event.preventDefault()
    const key = input.trim()
    if (!key) return

    setChecking(true)
    setError('')
    storeKey(key)
    try {
      await apiFetch('/detections', { params: { limit: 1 } })
      onUnlocked(key)
    } catch (err) {
      clearKey()
      setError(
        err.status === 401
          ? 'That key was rejected by the API.'
          : `Could not reach the API: ${err.message}`,
      )
    } finally {
      setChecking(false)
    }
  }

  return (
    <div className="keygate">
      <form className="keygate-card" onSubmit={handleSubmit}>
        <h1>Vision Classifier</h1>
        <p>Enter the API key to view detections.</p>
        <input
          type="password"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="API key"
          autoFocus
        />
        <button type="submit" disabled={checking || !input.trim()}>
          {checking ? 'Checking…' : 'Unlock'}
        </button>
        {error && <p className="keygate-error">{error}</p>}
      </form>
    </div>
  )
}
