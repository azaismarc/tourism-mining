import itertools
import csv

MODELS = ["all-mpnet-base-v2"]
GUIDES = [
    "all-mpnet-base-v2",
    "Alibaba-NLP/gte-modernbert-base",
    "vahidthegreat/StanceAware-SBERT"
]
LOSS = ["gist", "mnrl"]
LORA = [0, 8, 32]
DATASET = ["simcse", "senticse", "topicgpt/Llama-3.1-8B-Instruct/seed"]

SEEDS = [42, 128, 256]
BATCH = [32]
MARGIN = [0]

combinations = []

# Seed is now the OUTERMOST loop
for seed in SEEDS:

    # Generate all configurations for this seed
    for model, loss, lora, dataset, batch, guide, margin in itertools.product(
        MODELS,
        LOSS,
        LORA,
        DATASET,
        BATCH,
        GUIDES,
        MARGIN
    ):
        # Exclude simcse × gist
        if loss == "gist" and dataset == "simcse":
            continue

        # # MNRL only with all-mpnet guide
        if loss == "mnrl" and guide != "all-mpnet-base-v2":
            continue

        # Exclude modernbert configurations
        if "modernbert" in model and loss == "mnrl":
            continue

        if margin > 0 and loss == 'mnrl': continue

        combinations.append((
            model,
            loss,
            lora,
            dataset,
            seed,
            batch,
            guide,
            margin
        ))

# Write to TSV
with open("experiments.tsv", "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")

    writer.writerow([
        "model",
        "loss",
        "lora",
        "dataset",
        "seed",
        "batch",
        "guide",
        "margin"
    ])

    writer.writerows(combinations)

print(f"Generated experiments.tsv with {len(combinations)} combinations.")