cd /users/myacoubi/ComputeSpace/private/ondemand/data/sys/myjobs/13_11/FedPer_Flower_Complete
export PYTHONPATH=$(pwd)
export FL_SEED=0
mkdir -p results_multi_seed/wifi/seed_0/seed_0/logs

python -I scripts/run_server.py --rounds 2 --network wifi --address 0.0.0.0:8080 --min-clients 2 --local-epochs 3 --mu 0.01 --temp 0.5 --results-dir results_multi_seed/wifi/seed_0/seed_0 > results_multi_seed/wifi/seed_0/seed_0/logs/server.log 2>&1 &

sleep 5

for CID in 0 1 2 3; do
  USE_FOCAL_LOSS=0 FL_SEED=0 python -I scripts/run_client.py --client-id $CID --server-addr 127.0.0.1:8080 --network wifi --results-dir results_multi_seed/wifi/seed_0/seed_0 --data-dir data/client_datasets > results_multi_seed/wifi/seed_0/seed_0/logs/client_$CID.log 2>&1 &
done

echo "Seed 0 lancé — surveillez avec : tail -f results_multi_seed/wifi/seed_0/seed_0/logs/client_0.log"
wait
echo "=== SEED 0 TERMINÉ ==="