import sys
import time
import warnings

warnings.filterwarnings("ignore")
from huggingface_hub import snapshot_download

repo, dest = sys.argv[1], sys.argv[2]
for attempt in range(1, 41):
    try:
        p = snapshot_download(
            repo,
            local_dir=dest,
            allow_patterns=["*.safetensors", "*.json", "*.txt"],
            ignore_patterns=["original/*", "*/original/*", "consolidated*"],
            max_workers=2,
        )
        print("DONE", p, flush=True)
        break
    # Deliberately broad: any transient network or hub failure should retry.
    except Exception as e:  # noqa: BLE001
        print(f"retry {attempt}: {type(e).__name__}: {str(e)[:160]}", flush=True)
        time.sleep(min(10 * attempt, 60))
else:
    print("GAVEUP", flush=True)
