import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Per INTERFACE.md "Vite Config" — verbatim proxy table.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api':  { target: 'http://localhost:8000', changeOrigin: true },
      '/ws':   { target: 'ws://localhost:8000', ws: true },
      '/sim':  { target: 'http://localhost:8000', changeOrigin: true },
    }
  }
})
