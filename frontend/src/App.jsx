import { useCallback, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { clearKey, getStoredKey } from './api.js'
import KeyGate from './KeyGate.jsx'
import Dashboard from './Dashboard.jsx'
import './App.css'

export default function App() {
  const [apiKey, setApiKey] = useState(getStoredKey)
  const queryClient = useQueryClient()

  const handleUnlocked = useCallback(
    (key) => {
      queryClient.clear()
      setApiKey(key)
    },
    [queryClient],
  )

  const handleAuthError = useCallback(() => {
    clearKey()
    queryClient.clear()
    setApiKey('')
  }, [queryClient])

  if (!apiKey) {
    return <KeyGate onUnlocked={handleUnlocked} />
  }
  return <Dashboard onAuthError={handleAuthError} onLock={handleAuthError} />
}
