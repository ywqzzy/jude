#!/usr/bin/env bash
# ray-metrics-docker.sh — Prometheus + Grafana for the Ray Dashboard, as Docker
# containers (works with Colima). Points at the configs Ray auto-generates and
# rewrites its Prometheus service-discovery so containers can reach the host's
# Ray metrics endpoints via host.docker.internal.
#
# WHY: Ray's Dashboard "Metrics" tab needs Prometheus (scrapes Ray) + Grafana
# (renders). jude's OWN dashboard (:8477) needs none of this — this is only to
# light up the *Ray* dashboard's time-series charts.
#
# Prereqs: a running Docker daemon (Colima: `colima start`) and a running Ray
# cluster (so /tmp/ray/session_latest exists — e.g. `python -m jude.observe`).
#
# Usage:
#   benchmarking/ray-metrics-docker.sh up      # start both containers
#   benchmarking/ray-metrics-docker.sh down     # stop + remove
set -euo pipefail

RAY_METRICS="/tmp/ray/session_latest/metrics"
SD_SRC="/tmp/ray/prom_metrics_service_discovery.json"
# Colima shares $HOME by default but not always /tmp reliably; keep work under $HOME.
WORK="${HOME}/.jude/ray_metrics_docker"
NET="ray-metrics-net"
PROM_PORT=9090
GRAFANA_PORT=3000

up() {
  command -v docker >/dev/null || { echo "ERROR: docker not found" >&2; exit 1; }
  docker info >/dev/null 2>&1 || { echo "ERROR: docker daemon not running (try: colima start)" >&2; exit 1; }
  [[ -f "$RAY_METRICS/prometheus/prometheus.yml" ]] || {
    echo "ERROR: $RAY_METRICS not found — start a Ray cluster first (python -m jude.observe)" >&2
    exit 1
  }

  mkdir -p "$WORK"

  # 1) Rewrite Ray's service-discovery: 127.0.0.1 -> host.docker.internal so the
  #    containers can reach the host's Ray metrics exporters. Config + SD live in
  #    dedicated subdirs (Colima virtiofs can't bind-mount a single file, so we
  #    mount DIRECTORIES and reference the file inside).
  rm -rf "$WORK/prom" "$WORK/sd" "$WORK/grafana"
  mkdir -p "$WORK/prom" "$WORK/sd" "$WORK/grafana/datasources" "$WORK/grafana/dashboards"
  if [[ -f "$SD_SRC" ]]; then
    sed 's/127\.0\.0\.1/host.docker.internal/g' "$SD_SRC" > "$WORK/sd/sd.json"
  else
    echo '[]' > "$WORK/sd/sd.json"
    echo "WARN: $SD_SRC not found; Prometheus will have no Ray targets yet."
  fi

  # 2) Prometheus config that reads our rewritten SD file.
  cat > "$WORK/prom/prometheus.yml" <<'YML'
global:
  scrape_interval: 5s
  evaluation_interval: 5s
scrape_configs:
- job_name: 'ray'
  file_sd_configs:
  - files: ['/sd/sd.json']
YML

  # 3) Grafana provisioning: Prometheus datasource + point at the prom container.
  cat > "$WORK/grafana/datasources/ds.yml" <<'YML'
apiVersion: 1
datasources:
- name: Prometheus
  type: prometheus
  access: proxy
  url: http://ray-prometheus:9090
  isDefault: true
YML
  cat > "$WORK/grafana/dashboards/dash.yml" <<'YML'
apiVersion: 1
providers:
- name: 'ray'
  folder: 'Ray'
  type: file
  options:
    path: /ray-dashboards
YML
  # Copy Ray's generated Grafana dashboards into $WORK (under $HOME, which Colima
  # shares) so the container can read them even if /tmp isn't shared.
  mkdir -p "$WORK/ray-dashboards"
  cp -f "$RAY_METRICS/grafana/dashboards/"*.json "$WORK/ray-dashboards/" 2>/dev/null || true

  docker network create "$NET" >/dev/null 2>&1 || true

  echo "[prometheus] starting on :$PROM_PORT ..."
  docker rm -f ray-prometheus >/dev/null 2>&1 || true
  docker run -d --name ray-prometheus --network "$NET" \
    --add-host host.docker.internal:host-gateway \
    -p "$PROM_PORT:9090" \
    -v "$WORK/prom:/etc/prometheus:ro" \
    -v "$WORK/sd:/sd:ro" \
    prom/prometheus:latest >/dev/null

  echo "[grafana] starting on :$GRAFANA_PORT ..."
  docker rm -f ray-grafana >/dev/null 2>&1 || true
  docker run -d --name ray-grafana --network "$NET" \
    -p "$GRAFANA_PORT:3000" \
    -e GF_AUTH_ANONYMOUS_ENABLED=true \
    -e GF_AUTH_ANONYMOUS_ORG_ROLE=Admin \
    -e GF_SECURITY_ALLOW_EMBEDDING=true \
    -v "$WORK/grafana/datasources:/etc/grafana/provisioning/datasources:ro" \
    -v "$WORK/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro" \
    -v "$WORK/ray-dashboards:/ray-dashboards:ro" \
    grafana/grafana:latest >/dev/null

  sleep 3
  echo
  echo "Prometheus:  http://localhost:$PROM_PORT   (targets: http://localhost:$PROM_PORT/targets)"
  echo "Grafana:     http://localhost:$GRAFANA_PORT   (anonymous admin)"
  echo
  echo "Tell Ray where they are, THEN restart your Ray driver so the dashboard links them:"
  echo "  export RAY_PROMETHEUS_HOST=http://localhost:$PROM_PORT"
  echo "  export RAY_GRAFANA_HOST=http://localhost:$GRAFANA_PORT"
  echo "Then open the Ray Dashboard Metrics tab: http://127.0.0.1:8265/#/metrics"
  echo
  echo "NOTE: Ray's metrics ports change per session; re-run '$0 up' after restarting Ray."
}

down() {
  docker rm -f ray-prometheus ray-grafana >/dev/null 2>&1 || true
  docker network rm "$NET" >/dev/null 2>&1 || true
  echo "stopped ray-prometheus, ray-grafana"
}

case "${1:-up}" in
  up) up ;;
  down) down ;;
  *) echo "usage: $0 {up|down}" >&2; exit 1 ;;
esac
