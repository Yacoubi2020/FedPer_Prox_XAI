#!/bin/bash
# run_all_networks.sh — Lance toute l'expérience pour les 4 profils réseau.
# À exécuter sur le Pi SERVEUR (avec SSH passwordless vers les 4 clients).
#
# Usage:  bash scripts/run_all_networks.sh [ROUNDS]
#
set -e

# ════════ CONFIGURATION (à adapter) ════════
SERVER_IP="100.84.149.32"
SERVER_PORT="8080"
declare -a CLIENT_IPS=("100.115.141.83" "100.101.245.70" "100.88.9.122" "100.68.31.42")
declare -a CLIENT_USERS=("clientrasp2" "clientrasp3" "clientrasp4" "clientrasp5")
PROJECT_DIR="/home/clientrasp1/Desktop/FedPer_Flower_Complete"
VENV_ACTIVATE="source venv/bin/activate"
ROUNDS="${1:-30}"
NETWORKS=("wifi" "5g" "4g" "3g")
# ═══════════════════════════════════════════

mkdir -p logs results

for NET in "${NETWORKS[@]}"; do
    echo ""
    echo "╔═════════════════════════════════════════════════════════════╗"
    echo "║  ⚡ EXPÉRIENCE : ${NET^^}   ⏱  Rounds : $ROUNDS                  ║"
    echo "╚═════════════════════════════════════════════════════════════╝"
    mkdir -p results/$NET logs/$NET

    # 1) Serveur
    echo "🖥️  Démarrage serveur..."
    python scripts/run_server.py --rounds $ROUNDS --network $NET \
        --address 0.0.0.0:$SERVER_PORT --results-dir results/ \
        > logs/$NET/server.log 2>&1 &
    SERVER_PID=$!
    sleep 10

    # 2) Clients via SSH
    echo "📱 Lancement des 4 clients..."
    declare -a CLIENT_PIDS=()
    for i in 0 1 2 3; do
        USER="${CLIENT_USERS[$i]}"; HOST="${CLIENT_IPS[$i]}"
        ssh -o StrictHostKeyChecking=no ${USER}@${HOST} \
            "cd $PROJECT_DIR && $VENV_ACTIVATE && \
             python scripts/run_client.py --client-id $i \
                --server-addr ${SERVER_IP}:${SERVER_PORT} --network $NET \
                --results-dir results/ > logs/$NET/client_$i.log 2>&1" &
        CLIENT_PIDS+=($!)
        echo "  ✓ Client $i sur ${HOST}"
    done

    wait $SERVER_PID || true
    for pid in "${CLIENT_PIDS[@]}"; do wait $pid || true; done
    echo "✅ Training terminé pour $NET"

    # 3) SHAP sur chaque client
    echo "🔬 Analyse SHAP sur les 4 clients..."
    for i in 0 1 2 3; do
        USER="${CLIENT_USERS[$i]}"; HOST="${CLIENT_IPS[$i]}"
        ssh -o StrictHostKeyChecking=no ${USER}@${HOST} \
            "cd $PROJECT_DIR && $VENV_ACTIVATE && \
             python scripts/run_shap.py --client-id $i --network $NET \
                --results-dir results/ > logs/$NET/shap_$i.log 2>&1" &
    done
    wait

    # 4) Rapatrier les résultats
    echo "📥 Rapatriement résultats SHAP..."
    for i in 0 1 2 3; do
        USER="${CLIENT_USERS[$i]}"; HOST="${CLIENT_IPS[$i]}"
        mkdir -p results/$NET/client_$i
        scp -o StrictHostKeyChecking=no \
            ${USER}@${HOST}:$PROJECT_DIR/results/$NET/client_$i/shap_results.pkl \
            results/$NET/client_$i/ 2>/dev/null || echo "  ⚠️ pas de SHAP client $i"
        scp -o StrictHostKeyChecking=no \
            ${USER}@${HOST}:$PROJECT_DIR/results/$NET/client_$i/final_summary.json \
            results/$NET/client_$i/ 2>/dev/null || true
        scp -o StrictHostKeyChecking=no \
            ${USER}@${HOST}:$PROJECT_DIR/results/$NET/client_$i/metrics.json \
            results/$NET/client_$i/ 2>/dev/null || true
    done

    echo "✅ $NET terminé."; sleep 3
done

# 5) Génération des graphes finaux
echo ""
echo "╔═════════════════════════════════════════════════════════════╗"
echo "║  📊 GÉNÉRATION DES GRAPHES                                  ║"
echo "╚═════════════════════════════════════════════════════════════╝"
python scripts/visualize_all.py --all-networks --results-dir results/

echo ""
echo "🎉 EXPÉRIENCES TERMINÉES"
echo "📁 Résultats : results/{wifi,5g,4g,3g}/figures/"
echo "📊 Comparaison : results/comparison_figures/"
