import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The jude observability HTTP endpoint (jude.observe.serve) defaults to :8477.
// In dev we proxy /api to it so the dashboard and the metrics server can run on
// different ports without CORS friction.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5273,
    proxy: {
      "/api": {
        target: process.env.JUDE_METRICS_URL || "http://127.0.0.1:8477",
        changeOrigin: true,
      },
    },
  },
});
