#!/usr/bin/env bash
set -euo pipefail

IFACE="${1:-eno1}"
PC_IP="${PC_IP:-172.16.0.1}"
ROBOT_IP="${ROBOT_IP:-172.16.0.2}"
ROBOT_MAC="${ROBOT_MAC:-90:1B:0E:FD:8F:1D}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo:"
  echo "  sudo $0 ${IFACE}"
  exit 2
fi

if ! ip link show "${IFACE}" >/dev/null 2>&1; then
  echo "Network interface not found: ${IFACE}"
  ip -br link
  exit 2
fi

if ! command -v dnsmasq >/dev/null 2>&1; then
  echo "dnsmasq is missing. Install it first:"
  echo "  sudo apt update && sudo apt install -y dnsmasq-base"
  exit 2
fi

echo "Configuring ${IFACE} for Franka direct connection..."
ip link set "${IFACE}" up
ip addr add "${PC_IP}/24" dev "${IFACE}" 2>/dev/null || true

echo
echo "PC Franka-side address:"
ip -br addr show "${IFACE}"

echo
echo "Starting temporary DHCP server for Franka."
echo "Expected robot IP: ${ROBOT_IP}"
echo "Keep this terminal open. Stop it with Ctrl+C after testing."
echo

exec dnsmasq --no-daemon --log-dhcp \
  --port=0 \
  --interface="${IFACE}" --bind-interfaces --dhcp-authoritative \
  --listen-address="${PC_IP}" \
  --dhcp-range="${ROBOT_IP},172.16.0.20,255.255.255.0,12h" \
  --dhcp-host="${ROBOT_MAC},${ROBOT_IP}"
