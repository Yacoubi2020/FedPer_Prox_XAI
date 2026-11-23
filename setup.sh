#!/bin/bash
# Installation FedPer-Prox-XAI sur Raspberry Pi 5
set -e

echo "🚀 Installation FedPer-Prox-XAI Flower Complete"
echo ""

sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv python3-dev build-essential \
    cmake libopenblas-dev

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip wheel setuptools
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

mkdir -p data/client_datasets results logs

chmod +x scripts/*.py scripts/*.sh

echo ""
echo "✅ Installation terminée"
echo ""
echo "Étapes suivantes :"
echo "  1. source venv/bin/activate"
echo "  2. Pour test rapide : python scripts/generate_fake_data.py --samples 2000"
echo "  3. Préparer données  : python scripts/prepare_data.py --csv-dir data/"
echo "  4. Lancer serveur    : python scripts/run_server.py --rounds 5 --network 5g"
echo "  5. Lancer client     : python scripts/run_client.py --client-id N --server-addr IP:8080 --network 5g"
echo ""
echo "Pour TOUT lancer automatiquement (5G/4G/3G/WiFi):"
echo "  bash scripts/run_all_networks.sh 30"
