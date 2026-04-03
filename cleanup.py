import wandb
from datetime import datetime
api = wandb.Api()

# Replace with your entity/project
runs = api.runs("fice/fice")

threshold_date = datetime(2026, 1, 5)

count = 0
for run in runs:
    ts = run.created_at.split('.', 1)[0].rstrip('Z')
    run_date = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S')
    
    if run_date < threshold_date:
        print(run.name)
        for artifact in run.logged_artifacts():
            try:
                artifact.delete()
            except Exception as e:
                continue
        run.delete()
        count += 1
        if count % 10 == 0:
            print(f"Deleted {count} runs...")

print(f"Cleanup complete. Total deleted: {count}")