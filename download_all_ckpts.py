import subprocess
import os

def run_script(directory, script_name):
    script_path = os.path.join(directory, script_name)
    if os.path.exists(script_path):
        subprocess.run(["bash", script_name], cwd=directory, check=True)
    else:
        print(f"Script {script_path} not found!")

# Run the scripts in their respective directories
run_script('checkpoints', 'download_ckpts.sh')
run_script('gdino_checkpoints', 'download_ckpts.sh')
run_script('yolo_checkpoints', 'download_ckpts.sh')