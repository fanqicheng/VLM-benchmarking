import os
import argparse
import h5py
import numpy as np
import torch
import tempfile
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from trident import Processor
from trident.patch_encoder_models import CustomInferenceEncoder
from run_batch_of_slides import initialize_processor, run_task, build_parser
from load_model_trident import load_fm, FM_REGISTRY

UNITOPATHO_MPP = 0.44


class FMInferenceEncoder(CustomInferenceEncoder):
    def __init__(self, fm, **kwargs):
        super().__init__(**kwargs)
        self._encode_image = fm.encode_image

    def forward(self, x):
        return self._encode_image(x)


def get_encoder_name(args):
    if hasattr(args, 'model') and args.model:
        return args.model
    return args.patch_encoder


def make_mpp_csv(png_files: list) -> str:
    """Create a temporary CSV file with WSI names and mpp for Trident."""
    wsi_names = [f.name for f in png_files]
    df = pd.DataFrame({
        'wsi': wsi_names,
        'mpp': [UNITOPATHO_MPP] * len(wsi_names)
    })
    tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def merge_wsi_features(wsi_dir: str, job_dir: str, mag: float, patch_size: int, overlap: int, enc_name: str):
    """Merge features from all PNGs in a WSI folder into a single h5 file."""
    coords_subdir = f'{mag}x_{patch_size}px_{overlap}px_overlap'
    feat_dir = os.path.join(job_dir, coords_subdir, f'features_{enc_name}')

    wsi_name = os.path.basename(wsi_dir)
    merged_output_dir = os.path.join(job_dir, coords_subdir, f'features_{enc_name}_merged')
    merged_output = os.path.join(merged_output_dir, f'{wsi_name}.h5')
    os.makedirs(merged_output_dir, exist_ok=True)

    if os.path.exists(merged_output):
        print(f'[MERGE] {wsi_name}: already merged, skipping.')
        return

    png_names = [f.stem for f in Path(wsi_dir).glob('*.png')]
    feat_files = [os.path.join(feat_dir, f'{name}.h5') for name in png_names]
    feat_files = [f for f in feat_files if os.path.exists(f)]

    if not feat_files:
        print(f'[MERGE] No feature files found for {wsi_name}, skipping.')
        return

    all_features = []
    all_coords = []
    for feat_file in feat_files:
        with h5py.File(feat_file, 'r') as f:
            all_features.append(f['features'][:])
            if 'coords' in f:
                all_coords.append(f['coords'][:])

    merged_features = np.concatenate(all_features, axis=0)

    with h5py.File(merged_output, 'w') as f:
        f.create_dataset('features', data=merged_features)
        if all_coords:
            merged_coords = np.concatenate(all_coords, axis=0)
            f.create_dataset('coords', data=merged_coords)

    print(f'[MERGE] {wsi_name}: {merged_features.shape} → {merged_output}')


def run_feat_custom(processor, encoder, args):
    """Run feature extraction with custom FM encoder."""
    coords_dir = f'{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap'
    saveto = os.path.join(coords_dir, f'features_{encoder.enc_name}')
    processor.run_patch_feature_extraction_job(
        coords_dir=coords_dir,
        patch_encoder=encoder,
        device=f'cuda:{args.gpu}',
        saveas='h5',
        batch_limit=args.feat_batch_size or args.batch_size,
        saveto=saveto,
    )


def main():
    parser = build_parser()
    parser.add_argument('--model', type=str, default=None,
                        choices=list(FM_REGISTRY.keys()),
                        help='Custom FM model name (plip, keep, pathgen-clip, etc.)')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Local checkpoint path (required for mi-zero)')
    for action in parser._actions:
        if action.dest == 'patch_encoder':
            action.required = False
            action.default = None
    args = parser.parse_args()

    # UnitoPatho specific settings
    args.reader_type = 'image'
    args.search_nested = False

    use_custom = args.model is not None
    enc_name = get_encoder_name(args)

    # Load custom encoder once
    custom_encoder = None
    if use_custom:
        fm = load_fm(args.model, device=f'cuda:{args.gpu}',
                     **({'ckpt_path': args.ckpt_path} if args.ckpt_path else {}))
        custom_encoder = FMInferenceEncoder(
            fm=fm,
            enc_name=fm.name,
            model=fm.model,
            transforms=fm.preprocess,
            precision=fm.precision,
        )

    wsi_root = Path(args.wsi_dir)
    # two layers：parent → class folder (HP, NORM...) → wsi folder
    class_dirs = sorted([d for d in wsi_root.iterdir() if d.is_dir()])
    wsi_dirs = []
    for class_dir in class_dirs:
        for wsi_dir in sorted(class_dir.iterdir()):
            if wsi_dir.is_dir():
                wsi_dirs.append(wsi_dir)
    print(f'Found {len(wsi_dirs)} WSI folders across {len(class_dirs)} classes in {wsi_root}')

    original_job_dir = args.job_dir
    original_task = args.task

    for wsi_dir in tqdm(wsi_dirs, desc='Processing WSIs'):
        png_files = list(wsi_dir.glob('*.png'))
        if not png_files:
            print(f'[SKIP] No PNG files in {wsi_dir.name}')
            continue

        print(f'\n[WSI] {wsi_dir.name} — {len(png_files)} regions')

        # create mpp csv 
        tmp_csv = make_mpp_csv(png_files)

        args.wsi_dir = str(wsi_dir)
        args.job_dir = os.path.join(original_job_dir, wsi_dir.parent.name, wsi_dir.name)
        args.custom_list_of_wsis = tmp_csv
        args.task = original_task

        processor = initialize_processor(args)
        tasks = ['seg', 'coords', 'feat'] if args.task == 'all' else [args.task]

        for task_name in tasks:
            args.task = task_name
            if task_name == 'feat' and use_custom:
                run_feat_custom(processor, custom_encoder, args)
            else:
                run_task(processor, args)
              
        os.unlink(tmp_csv)

        # merge features
        if 'feat' in tasks or original_task == 'all':
            merge_wsi_features(
                wsi_dir=str(wsi_dir),
                job_dir=args.job_dir,
                mag=args.mag,
                patch_size=args.patch_size,
                overlap=args.overlap,
                enc_name=enc_name,
            )

    args.job_dir = original_job_dir
    args.task = original_task


if __name__ == '__main__':
    main()
