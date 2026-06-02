import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        timeout: 0,          // no proxy-level timeout — required for long SSE streams
        proxyTimeout: 0,     // no upstream timeout — deepthink can take 60-120s
        configure: (proxy) => {
          // Disable browser-side socket timeout so SSE connections stay open
          proxy.on('proxyReq', (_proxyReq, _req, res) => {
            res.socket?.setTimeout(0)
          })
          proxy.on('error', (err, _req, res) => {
            console.error('[vite proxy]', err.message)
            if (res && !res.headersSent) {
              res.writeHead(503, { 'Content-Type': 'text/plain' })
              res.end('Backend unreachable: ' + err.message)
            }
          })
        },
      },
    },
  },
})
