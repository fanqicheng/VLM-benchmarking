from trident.patch_encoder_models import CustomInferenceEncoder
from load_model_trident import load_fm, FM_REGISTRY
from run_batch_of_slides import build_parser, initialize_processor, run_task


class FMInferenceEncoder(CustomInferenceEncoder):
    """覆写 forward，用 encode_image 而不是直接调用 model"""
    def __init__(self, fm, **kwargs):
        super().__init__(**kwargs)
        self._encode_image = fm.encode_image

    def forward(self, x):
        return self._encode_image(x)


def main():
    parser = build_parser()
    parser.add_argument('--model', type=str, required=True,
                        choices=list(FM_REGISTRY.keys()),
                        help='FM model name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Local checkpoint path for models that require it (e.g. mi-zero)')
    for action in parser._actions:
        if action.dest == 'patch_encoder':
            action.choices = None
            action.default = 'custom'
            action.required = False
    args = parser.parse_args()

    fm = load_fm(args.model, device=f'cuda:{args.gpu}', 
             **({'ckpt_path': args.ckpt_path} if args.ckpt_path else {}))

    encoder = FMInferenceEncoder(
        fm=fm,
        enc_name=fm.name,
        model=fm.model,
        transforms=fm.preprocess,
        precision=fm.precision,
    )

    processor = initialize_processor(args)
    tasks = ['seg', 'coords', 'feat'] if args.task == 'all' else [args.task]

    for task_name in tasks:
        args.task = task_name
        if task_name == 'feat':
            coords_dir = args.coords_dir or f'{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap'
            processor.run_patch_feature_extraction_job(
                coords_dir=coords_dir,
                patch_encoder=encoder,
                device=f'cuda:{args.gpu}',
                saveas='h5',
                batch_limit=args.feat_batch_size or args.batch_size,
            )
        else:
            run_task(processor, args)

if __name__ == '__main__':
    main()