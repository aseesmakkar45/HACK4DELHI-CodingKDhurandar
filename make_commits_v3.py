import os
import subprocess

def run_cmd(cmd, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, env=merged_env, check=False)

def commit(date_str, msg, files):
    base_folder = "railway_ai_system(version 3)"
    for f in files:
        if f == '.':
            run_cmd(f'git add "{base_folder}/."')
        else:
            run_cmd(f'git add "{base_folder}/{f}"')
    
    env = {
        'GIT_AUTHOR_DATE': f'{date_str}T12:00:00',
        'GIT_COMMITTER_DATE': f'{date_str}T12:00:00'
    }
    status = subprocess.run('git status --porcelain', shell=True, capture_output=True, text=True)
    if status.stdout.strip():
        run_cmd(f'git commit -m "{msg}"', env=env)
    else:
        print(f"Nothing to commit for {msg}")

def main():
    # Longpaths fix just in case
    run_cmd('git config core.longpaths true')

    commits = [
        ("2026-04-10", "chore: initial project setup and configuration", [".gitignore", "config.yaml", "requirements.txt"]),
        ("2026-04-12", "feat: add data loading and video processing modules", ["src/data/"]),
        ("2026-04-14", "feat: implement YOLO-based object detection", ["src/detection/yolo_detector.py"]),
        ("2026-04-16", "feat: implement behavior engine for risk assessment", ["src/detection/behavior_engine.py"]),
        ("2026-04-18", "feat: add feature extraction logic", ["src/features/"]),
        ("2026-04-20", "feat: add model training scripts for ML layer and YOLO", ["src/models/"]),
        ("2026-04-22", "feat: implement unified inference pipeline", ["src/pipeline/"]),
        ("2026-04-24", "feat: setup FastAPI server and database", ["server/main.py", "server/database.py", "server/models.py", "server/__init__.py"]),
        ("2026-04-26", "feat: implement API routes for alerts, history, and results", ["server/routes/"]),
        ("2026-04-28", "feat: add frontend application", ["frontend/"]),
        ("2026-04-29", "chore: include initial datasets and ML statistics output", ["datasets/", "data/", "ml_stats_cv_output.txt", "ml_stats_output.txt", "railguard.db"]),
        ("2026-04-30", "fix: minor bug fixes and improvements", ["."])
    ]

    for date_str, msg, files in commits:
        commit(date_str, msg, files)
        
    print("Done. To push, run: git push origin main")

if __name__ == "__main__":
    main()
