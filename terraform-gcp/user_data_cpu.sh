#!/bin/bash
set -euo pipefail

exec > >(tee /var/log/user-data.log | logger -t startup-script -s 2>/dev/console) 2>&1
export DEBIAN_FRONTEND=noninteractive

echo "Starting startup script for CPU LightGBM benchmark node"

apt-get update -y
apt-get install -y --no-install-recommends ca-certificates python3 python3-pip

# Debian 12 marks the system Python as externally managed (PEP 668);
# --break-system-packages is required for a system-wide pip install here.
python3 -m pip install --break-system-packages --disable-pip-version-check --upgrade pip
python3 -m pip install --break-system-packages --disable-pip-version-check \
  lightgbm scikit-learn pandas numpy kaggle

echo "CPU environment ready: lightgbm, scikit-learn, pandas, numpy, kaggle installed system-wide."
