Trained weights are saved here by `scripts/train.py`:

    models/<run_name>/final_model.zip   final policy (commit this if you want to keep it in the repo)
    models/<run_name>/best_model.zip    best policy seen by the periodic evaluation
    models/<run_name>/checkpoints/      periodic snapshots (git-ignored)
    models/<run_name>/config.json       env / reward / train config used
