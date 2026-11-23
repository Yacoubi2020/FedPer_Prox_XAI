#!/bin/bash
# run_all_networks_local.sh — Version LOCAL avec monitoring temps réel.
# Affiche en direct la progression (loss, acc, round courant) au lieu
# de rester figé sur "Training en cours...".
#
# Usage:
#   bash scripts/run_all_networks_local.sh [ROUNDS] [LOCAL_EPOCHS]
#
# Exemples :
#   bash scripts/run_all_networks_local.sh 5       # 5 rounds × 3 epochs
#   bash scripts/run_all_networks_local.sh 5 1     # 5 rounds × 1 epoch (3× plus rapide)
#   bash scripts/run_all_networks_local.sh 3 1     # test très rapide
#

# ═══════════════ CONFIGURATION ═══════════════
SERVER_PORT="8080"
ROUNDS="${1:-5}"
LOCAL_EPOCHS="${2:-3}"
NETWORKS=("wifi" "5g" "4g" "3g")
# ════════════════════════════════════════════

mkdir -p logs results

# ─── Helpers ──────────────────────────────────────────────────────────────

cleanup_all() {
    pkill -9 -f "run_server" 2>/dev/null
    pkill -9 -f "run_client" 2>/dev/null
    pkill -9 -f "run_shap" 2>/dev/null
    sleep 2
}

# Affiche live la progression du serveur tant qu'il tourne
monitor_live() {
    local NET=$1
    local SRV_PID=$2
    local START=$(date +%s)
    local last_line=""

    while kill -0 $SRV_PID 2>/dev/null; do
        # Récupérer la dernière info utile
        local round_info=$(grep "📊 Round" logs/$NET/server.log 2>/dev/null | tail -1)
        local client_info=$(grep "loss=" logs/$NET/client_0.log 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//')
        local elapsed=$(( $(date +%s) - START ))
        local mm=$(( elapsed / 60 ))
        local ss=$(( elapsed % 60 ))

        # Choisir la ligne à afficher (priorité au round_info)
        local display=""
        if [ -n "$round_info" ]; then
            display="$round_info"
        elif [ -n "$client_info" ]; then
            display="C0: $client_info"
        else
            display="initialisation..."
        fi

        if [ "$display" != "$last_line" ]; then
            printf "\r\033[K  ⏱  [%02d:%02d] %s" $mm $ss "$display"
            last_line="$display"
        else
            printf "\r\033[K  ⏱  [%02d:%02d] %s" $mm $ss "$display"
        fi

        sleep 5
    done
    echo ""
}

# ─── Vérifs ───────────────────────────────────────────────────────────────

echo ""
echo "╔═════════════════════════════════════════════════════════════════╗"
echo "║  🔧 FedPer-Prox-XAI Flower - Expérience LOCALE                  ║"
printf "║  Rounds : %-3d   Epochs locaux : %-3d   Réseaux : 4              ║\n" $ROUNDS $LOCAL_EPOCHS
echo "╚═════════════════════════════════════════════════════════════════╝"

if [ ! -d "data/client_datasets/client_0" ]; then
    echo "❌ Partitions manquantes. Exécutez :"
    echo "   python scripts/prepare_data.py --csv-dir data/"
    exit 1
fi

echo ""
echo "📊 Données :"
for i in 0 1 2 3; do
    if [ -f "data/client_datasets/client_$i/X.npy" ]; then
        SAMPLES=$(python -c "import numpy as np; print(np.load('data/client_datasets/client_$i/X.npy').shape[0])" 2>/dev/null || echo "?")
        echo "  ↳ Client $i : $SAMPLES samples"
    fi
done

echo ""
echo "🧹 Nettoyage..."
cleanup_all

# ─── Boucle principale ────────────────────────────────────────────────────

GLOBAL_START=$(date +%s)
declare -a STATUS=()

for NET in "${NETWORKS[@]}"; do
    NET_START=$(date +%s)
    echo ""
    echo "╔═════════════════════════════════════════════════════════════════╗"
    printf "║  ⚡ RÉSEAU : %-7s   Rounds : %d   Epochs : %d                   ║\n" "${NET^^}" $ROUNDS $LOCAL_EPOCHS
    echo "╚═════════════════════════════════════════════════════════════════╝"

    mkdir -p results/$NET logs/$NET
    rm -f logs/$NET/server.log logs/$NET/client_*.log

    # 1) Serveur
    echo "🖥️  Démarrage serveur..."
    python -u scripts/run_server.py \
        --rounds $ROUNDS --network $NET \
        --address 0.0.0.0:$SERVER_PORT \
        --local-epochs $LOCAL_EPOCHS \
        --results-dir results/ \
        > logs/$NET/server.log 2>&1 &
    SERVER_PID=$!
    sleep 8

    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ Serveur crashé. Logs :"
        tail -15 logs/$NET/server.log
        STATUS+=("$NET:FAILED")
        continue
    fi
    echo "  ✓ Serveur PID $SERVER_PID OK"

    # 2) Clients
    echo "📱 Lancement de 4 clients..."
    declare -a CLIENT_PIDS=()
    for i in 0 1 2 3; do
        python -u scripts/run_client.py --client-id $i \
            --server-addr 127.0.0.1:${SERVER_PORT} --network $NET \
            --results-dir results/ \
            > logs/$NET/client_$i.log 2>&1 &
        CLIENT_PIDS+=($!)
    done
    echo "  ✓ 4 clients lancés (PIDs: ${CLIENT_PIDS[*]})"

    # 3) Monitoring temps réel
    echo ""
    echo "⏳ Entraînement (affichage live, Ctrl+C pour arrêter) :"
    monitor_live "$NET" $SERVER_PID

    # 4) Attente clients
    for pid in "${CLIENT_PIDS[@]}"; do
        wait $pid 2>/dev/null
    done
    NET_TIME=$(( $(date +%s) - NET_START ))
    echo "✅ Training $NET terminé en $((NET_TIME / 60))m$((NET_TIME % 60))s"

    # 5) Récap métriques
    echo ""
    echo "📊 Derniers rounds :"
    grep "📊 Round" logs/$NET/server.log | tail -3 | sed 's/^/  /'

    # 6) SHAP
    echo ""
    echo "🔬 Calcul SHAP (4 clients en parallèle)..."
    SHAP_START=$(date +%s)
    for i in 0 1 2 3; do
        python -u scripts/run_shap.py --client-id $i --network $NET \
            --results-dir results/ \
            > logs/$NET/shap_$i.log 2>&1 &
    done
    wait
    SHAP_TIME=$(( $(date +%s) - SHAP_START ))
    SHAP_OK=0
    for i in 0 1 2 3; do
        [ -f "results/$NET/client_$i/shap_results.pkl" ] && SHAP_OK=$((SHAP_OK + 1))
    done
    echo "✅ SHAP : $SHAP_OK/4 clients OK en $((SHAP_TIME / 60))m$((SHAP_TIME % 60))s"

    STATUS+=("$NET:OK")
    cleanup_all
done

# ─── Graphes ──────────────────────────────────────────────────────────────

echo ""
echo "╔═════════════════════════════════════════════════════════════════╗"
echo "║  📊 GÉNÉRATION DES GRAPHES                                      ║"
echo "╚═════════════════════════════════════════════════════════════════╝"
python scripts/visualize_all.py --all-networks --results-dir results/ 2>&1 | tail -25

# ─── Résumé ───────────────────────────────────────────────────────────────

TOTAL=$(( $(date +%s) - GLOBAL_START ))
echo ""
echo "╔═════════════════════════════════════════════════════════════════╗"
echo "║  🎉 EXPÉRIENCES TERMINÉES                                       ║"
echo "╚═════════════════════════════════════════════════════════════════╝"
echo "⏱  Temps total : $((TOTAL / 60))m$((TOTAL % 60))s"
echo ""
echo "📋 Bilan :"
for entry in "${STATUS[@]}"; do
    NET="${entry%:*}"; ST="${entry#*:}"
    case "$ST" in
        OK)     echo "  ✅ $NET" ;;
        FAILED) echo "  ❌ $NET (crash)" ;;
        *)      echo "  ❓ $NET ($ST)" ;;
    esac
done
echo ""
echo "📁 Résultats : results/{wifi,5g,4g,3g}/figures/"
echo "📊 Comparaison : results/comparison_figures/"
echo ""
echo "Voir les graphes :"
echo "  ls -la results/*/figures/ results/comparison_figures/"