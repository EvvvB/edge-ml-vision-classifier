import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import './index.css'
import App from './App.jsx'
import { isLocalEnvironment } from './env.js'

if (isLocalEnvironment) {
  document.documentElement.dataset.env = 'local'
  document.title = 'Vision Classifier (local)'
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // A 401 means the key is wrong; retrying will never fix it.
      retry: (failureCount, error) => error?.status !== 401 && failureCount < 2,
      staleTime: 15_000,
    },
  },
})

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
)
