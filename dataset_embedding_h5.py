import argparse
import numpy as np
import h5py
from pathlib import Path

parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, required=True,
#                     help='Model name to determine aggregation method')
parser.add_argument('--h5_dir', type=str, required=True,
                    help='Input directory containing h5 files')
parser.add_argument('--out_dir', type=str, required=True,
                    help='Output directory for embeddings')
parser.add_argument('--top_k', type=float, default=0.05,
                    help='Fraction of top patches to select by norm (e.g. 0.05 = 5%)')
args = parser.parse_args()

method_name = 'topk'

h5_dir = Path(args.h5_dir)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

TOP_K = args.top_k
out_name = h5_dir.name

# top k percentage 
agg_fn = lambda emb: emb[np.linalg.norm(emb, axis=1).argsort()[-max(1, int(len(emb) * TOP_K)):]].mean(axis=0)

embs, ids = [], []
for f in sorted(h5_dir.glob("*.h5")):
    slide_id = f.stem
    with h5py.File(f, 'r') as h5:
        features = h5['features'][:]
    if features.ndim == 2:
        features = agg_fn(features)
    embs.append(features)
    ids.append(slide_id)

np.save(out_dir / f'{out_name}_{method_name}.npy', np.stack(embs))
np.save(out_dir / f'{out_name}_{method_name}_ids.npy', np.array(ids))
print(f"[OK] agg={method_name}: saved {len(embs)} slides.")


