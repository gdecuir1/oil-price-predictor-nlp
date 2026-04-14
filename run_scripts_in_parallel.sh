#!/bin/bash

# ==========================================
# Cluster Scrape Deployment Configuration
# ==========================================
USERNAME="lhak"
TOTAL_NODES=22

# ==========================================
# STRATEGIC FIX: Unique GoLogin Profiles
# You must create 22 profiles in GoLogin and paste their IDs below.
# ==========================================
GOLOGIN_PROFILES=(
	"69d5c28dee87ea24b3752e14" # Node 1
	"69d5c14d7601e81f02e6cb11" # Node 2
	"69d5c1c08d15ccb26ac77c67" # Node 3
	"69d5c2247601e81f02e74e20" # Node 4
	"69d5c5352bb96f7059ae690d" # Node 5
	"69d5c566ddef7f8118005b88" # Node 6
)

# Ensure this path perfectly matches where the code lives on the ugrad machines
TARGET_DIR="~/codes/quant/data_mining_project/dataset_build/we_scrape_all_day"

# If your ugrad machines start at ugrad1, ugrad2, etc.
START_MACHINE_NUM=1

# Use absolute path to Conda python if standard python3 doesn't have your packages
PYTHON_CMD="python3"
# ==========================================

# Safety Check
if [ ${#GOLOGIN_PROFILES[@]} -ne $TOTAL_NODES ]; then
	echo "❌ ERROR: You requested $TOTAL_NODES nodes, but only provided ${#GOLOGIN_PROFILES[@]} GoLogin profiles."
	echo "Please ensure the GOLOGIN_PROFILES array has exactly $TOTAL_NODES entries."
	exit 1
fi

echo "🚀 Initiating deployment to $TOTAL_NODES ugrad machines..."

for ((i = 0; i < $TOTAL_NODES; i++)); do
	MACHINE_NUM=$((START_MACHINE_NUM + i))
	HOST="${USERNAME}@ugrad${MACHINE_NUM}.cs.jhu.edu"
	PROFILE_ID=${GOLOGIN_PROFILES[$i]}

	echo "📦 Dispatching Node Index $i to $HOST with Profile: $PROFILE_ID"

	ssh -o StrictHostKeyChecking=no "$HOST" "
        source ~/.bashrc; 
        
        cd ${TARGET_DIR} || exit;
        
        tmux kill-session -t scraper_node_${i} 2>/dev/null;
        
        # INJECTED THE -g FLAG WITH THE UNIQUE PROFILE ID
        tmux new-session -d -s scraper_node_${i} 'nohup ${PYTHON_CMD} scraper_engine.py -i tasks.json -g ${PROFILE_ID} --total-nodes ${TOTAL_NODES} --node-index ${i} > node_${i}.log 2>&1'
    "

	sleep 1
done

echo "✅ All $TOTAL_NODES jobs have been successfully dispatched!"
echo "To monitor a node in real-time, SSH into it and run: tail -f ${TARGET_DIR}/node_<index>.log"
