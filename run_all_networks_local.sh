#!/bin/bash
# run_all_networks_local.sh — Version LOCAL pour tester sur UNE SEULE machine
# (4 clients + 1 serveur sur la même machine via localhost).
# Idéal pour valider que tout fonctionne avant de déployer sur les 5 Pi.
#
# Usage:  bash scripts/run_all_networks_local.sh [ROUNDS]
#
set -e

# ════════ CONFIGURATION ════════
SERVER_PORT="8080"
ROUNDS="${1:-5}"           # Par défaut 5 rounds pour test rapide
NETWORKS=("wifi" "5g" "4g" "3g")
# ═══════════════════════════════

mkdir -p logs results

# Nettoyer toute exécution précédente
echo "🧹 Nettoyage des anciens processus..."
pkill -9 -f "run_server\|run_client" 2>/dev/null || true
sleep 2

for NET in "${NETWORKS[@]}"; do
    echo ""
    echo "╔═════════════════════════════════════════════════════════════╗"
    echo "║  ⚡ EXPÉRIENCE LOCALE : ${NET^^}   ⏱  Rounds : $ROUNDS              ║"
    echo "╚═════════════════════════════════════════════════════════════╝"
    mkdir -p results/$NET logs/$NET

    # 1) Démarrer le serveur en arrière-plan
    echo "🖥️  Démarrage serveur sur localhost:${SERVER_PORT}..."
    python scripts/run_server.py --rounds $ROUNDS --network $NET \
        --address 0.0.0.0:$SERVER_PORT --results-dir results/ \
        > logs/$NET/server.log 2>&1 &
    SERVER_PID=$!
    sleep 8

    # Vérifier que le serveur tourne
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ Le serveur a crashé. Logs :"
        tail -20 logs/$NET/server.log
        exit 1
    fi
    echo "  ✓ Serveur démarré (PID $SERVER_PID)"

    # 2) Lancer les 4 clients localement
    echo "📱 Lancement des 4 clients..."
    declare -a CLIENT_PIDS=()
    for i in 0 1 2 3; do
        python scripts/run_client.py --client-id $i \
            --server-addr 127.0.0.1:${SERVER_PORT} --network $NET \
            --results-dir results/ > logs/$NET/client_$i.log 2>&1 &
        CLIENT_PIDS+=($!)
        echo "  ✓ Client $i lancé (PID ${CLIENT_PIDS[$i]})"
    done

    # 3) Attendre la fin du serveur (= fin du training)
    echo "⏳ Training en cours..."
    wait $SERVER_PID || true
    echo "✅ Serveur terminé. Attente des clients..."
    for pid in "${CLIENT_PIDS[@]}"; do
        wait $pid 2>/dev/null || true
    done
    echo "✅ Training $NET terminé."

    # 4) SHAP local sur chaque client
    echo "🔬 Analyse SHAP locale..."
    for i in 0 1 2 3; do
        python scripts/run_shap.py --client-id $i --network $NET \
            --results-dir results/ \
            > logs/$NET/shap_$i.log 2>&1 &
    done
    wait
    echo "✅ SHAP $NET terminé."

    # Petite pause entre les expériences
    sleep 3
done

# 5) Génération des graphes finaux
echo ""
echo "╔═════════════════════════════════════════════════════════════╗"
echo "║  📊 GÉNÉRATION DES GRAPHES                                  ║"
echo "╚═════════════════════════════════════════════════════════════╝"
python scripts/visualize_all.py --all-networks --results-dir results/

echo ""
echo "╔═════════════════════════════════════════════════════════════╗"
echo "║  🎉 EXPÉRIENCES LOCALES TERMINÉES                           ║"
echo "╚═════════════════════════════════════════════════════════════╝"
echo "📁 Résultats par réseau : results/{wifi,5g,4g,3g}/figures/"
echo "📊 Comparaison globale  : results/comparison_figures/"
echo ""
echo "Pour visualiser :"
echo "  ls results/*/figures/"