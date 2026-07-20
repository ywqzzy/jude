#!/usr/bin/env bash
# ray-metrics-stack.sh — stand up Prometheus + Grafana for the Ray Dashboard's
# time-series charts, pointed at the configs Ray auto-generates.
#
# WHY: Ray's Dashboard "Metrics" tab needs a Prometheus (scrapes Ray) + Grafana
# (renders) running. Ray writes ready-made configs under
# /tmp/ray/session_latest/metrics; this script just launches the two servers
# against them. (jude's OWN dashboard at :8477 needs none of this.)
#
# Prereqs (macOS / Homebrew):  brew install prometheus grafana
# A Ray cluster must be running (so /tmp/ray/session_latest exists).
#
# Usage:
#   benchmarking/ray-metrics-stack.sh start   # launch both, print URLs
#   benchmarking/ray-metrics-stack.sh stop    # kill both
set -euo pipefail

RAY_METRICS="/tmp/ray/session_latest/metrics"
PROM_CFG="$RAY_METRICS/prometheus/prometheus.yml"
GRAFANA_HOME="${GRAFANA_HOMEPATH:-/opt/homebrew/opt/grafana/share/grafana}"
GRAFANA_INI="$RAY_METRICS/grafana/grafana.ini"
PROM_PORT=9090
GRAFANA_PORT=3000
PIDDIR="/tmp/ray_metrics_stack"
mkdir -p "$PIDDIR"

start() {
  if [[ ! -f "$PROM_CFG" ]]; then
    echo "ERROR: $PROM_CFG not found — is a Ray cluster running?" >&2
    echo "  Start one first, e.g.:  python -m jude.observe   (or ray.init() in your script)" >&2
    exit 1
  fi

  if command -v prometheus >/dev/null 2>&1; then
    echo "[prometheus] starting on :$PROM_PORT with Ray's config ..."
    prometheus --config.file="$PROM_CFG" --web.listen-address=":$PROM_PORT" \
      > "$PIDDIR/prometheus.log" 2>&1 &
    echo $! > "$PIDDIR/prometheus.pid"
  else
    echo "WARN: prometheus not on PATH — install with 'brew install prometheus'"
  fi

  if command -v grafana-server >/dev/null 2>&1; then
    echo "[grafana] starting on :$GRAFANA_PORT with Ray's config ..."
    # Ray's grafana.ini provisions the Prometheus datasource + Ray dashboards.
    grafana-server --config="$GRAFANA_INI" --homepath="$GRAFANA_HOME" \
      > "$PIDDIR/grafana.log" 2>&1 &
    echo $! > "$PIDDIR/grafana.pid"
  else
    echo "WARN: grafana-server not on PATH — install with 'brew install grafana'"
  fi

  sleep 2
  echo
  echo "Prometheus:  http://localhost:$PROM_PORT"
  echo "Grafana:     http://localhost:$GRAFANA_PORT  (default admin/admin)"
  echo "Now reload the Ray Dashboard Metrics tab: http://127.0.0.1:8265/#/metrics"
  echo "(If Ray uses non-default ports, set env RAY_PROMETHEUS_HOST / RAY_GRAFANA_HOST before ray.init.)"
}

stop() {
  for svc in prometheus grafana; do
    if [[ -f "$PIDDIR/$svc.pid" ]]; then
      kill "$(cat "$PIDDIR/$svc.pid")" 2>/dev/null || true
      rm -f "$PIDDIR/$svc.pid"
      echo "[$svc] stopped"
    fi
  done
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  *) echo "usage: $0 {start|stop}" >&2; exit 1 ;;
esac
