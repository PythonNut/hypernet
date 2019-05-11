import time
import shutil
import traceback

from pathlib import Path

from ensemble_stats import *

args = vars(load_args())
pt = Path(args['pt'])
last_mtime = -1
best_score = 0

while True:
    try:
        mtime = pt.stat().st_mtime
        if mtime != last_mtime:
            last_mtime = mtime
            _, ensemble_score, _, _ = main(**vars(load_args()))
            if ensemble_score > best_score:
                print("New best ensemble found!")
                best_score = ensemble_score
                shutil.copy(str(pt), str(pt.parent / 'ensemble_auto_checkpoint.pt'))
    except:
        traceback.print_exc()

    time.sleep(60)
