import numpy as np
import torch
import torch.nn.functional as F
import argparse
import pandas as pd
import re
import json
from pathlib import Path


def evaluate_ranking(query_embs, db_embs, db_labels, true_labels, normalize=True):
    if isinstance(query_embs, np.ndarray):
        query_embs = torch.tensor(query_embs, dtype=torch.float32)
    if isinstance(db_embs, np.ndarray):
        db_embs = torch.tensor(db_embs, dtype=torch.float32)

    if normalize:
        query_embs = F.normalize(query_embs, dim=-1)
        db_embs = F.normalize(db_embs, dim=-1)

    sims = query_embs @ db_embs.T

    # Ranking
    ranking = torch.argsort(sims, dim=1, descending=True).cpu().numpy()

    ranks = []
    for i in range(len(query_embs)):
        true_label = true_labels[i]
        retrieved = ranking[i]

        # remove itself
        retrieved = retrieved[retrieved != i]

        match_positions = np.where(db_labels[retrieved] == true_label)[0]
        if len(match_positions) == 0:
            ranks.append(1e9)
        else:
            ranks.append(match_positions[0])

    ranks = np.array(ranks)

    R1 = np.mean((ranks < 1).astype(float))
    R5 = np.mean((ranks < 5).astype(float))
    MRR = np.mean(1.0 / (ranks + 1))
    nDCG = np.mean([(1.0 / np.log2(r + 2)) if r < 5 else 0.0 for r in ranks])

    print("=========== RETRIEVAL METRICS ===========")
    print(f"Recall@1 : {R1:.4f}")
    print(f"Recall@5 : {R5:.4f}")
    print(f"MRR      : {MRR:.4f}")
    print(f"nDCG@5   : {nDCG:.4f}")
    print("==========================================")

    return sims.cpu().numpy(), ranking, {
        "recall@1": R1,
        "recall@5": R5,
        "mrr": MRR,
        "ndcg@5": nDCG
    }



def extract_tcga_case(text):
    m = re.match(r"(TCGA-\d{2}-\d{4})", text)
    return m.group(1) if m else None

