import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In dev, API calls are proxied to the local cloud-api so the app can use
// same-origin relative paths in every environment.
export default defineConfig({
  plugins: [react()],
  build: {
    // The API container serves the built app from cloud-api/static.
    outDir: '../cloud-api/static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/detections': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
