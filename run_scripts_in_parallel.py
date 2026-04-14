import subprocess
import sys
import time

# ==========================================
# Local Cluster Configuration
# ==========================================
# Define the subset of GoLogin Profiles you want to run LOCALLY.
# Note: Keep this number low (3 to 6) depending on your computer's RAM.
GOLOGIN_PROFILES = [
    "69d5cb014ebbb920595e2c1d",  # Node 0
    "69d5c14d7601e81f02e6cb11",  # Node 1
    "69d5c1c08d15ccb26ac77c67",  # Node 2
    "69d5c2247601e81f02e74e20",  # Node 3
    "69d5c5352bb96f7059ae690d",  # Node 4
    "69d5c566ddef7f8118005b88",  # Node 5
]

TASKS_FILE = "tasks.json"
# ==========================================


def launch_cluster():
    total_nodes = len(GOLOGIN_PROFILES)
    print(f"🚀 Initiating local deployment of {total_nodes} scraper nodes...")

    processes = []

    try:
        for i, profile_id in enumerate(GOLOGIN_PROFILES):
            print(f"📦 Booting Node {i} with Profile {profile_id}...")

            # Form the command. sys.executable ensures it uses your active Python/Conda environment
            cmd = [
                sys.executable,
                "scraper_engine.py",
                "-i",
                TASKS_FILE,
                "-g",
                profile_id,
                "-tn",
                str(total_nodes),
                "-ni",
                str(i),
            ]

            # Launch as an independent background process and log to a file
            log_file = open(f"node_{i}.log", "w")
            p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

            processes.append((p, log_file))
            time.sleep(2)  # Stagger boot times so GoLogin SDK initializes smoothly

        print("\n✅ All nodes are running in the background!")
        print(
            "To view live logs for a node, open a new terminal and run: tail -f node_0.log"
        )
        print("Press Ctrl+C here to gracefully shutdown the entire cluster.\n")

        # Keep the main thread alive waiting for processes to finish
        for p, _ in processes:
            p.wait()

    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C detected. Shutting down the cluster gracefully...")
        for p, log_file in processes:
            p.terminate()
            log_file.close()
        print("Cluster shutdown complete.")


if __name__ == "__main__":
    launch_cluster()
