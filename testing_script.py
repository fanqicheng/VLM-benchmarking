# test_models.py
import torch
from PIL import Image
from load_model_trident import FM_REGISTRY, load_fm
from trident.patch_encoder_models import CustomInferenceEncoder

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 造一张假图
dummy_image = Image.fromarray(
    (torch.rand(224, 224, 3) * 255).byte().numpy()
)

results = {}

for model_name in ['plip', 'keep', 'patho-clip', 'pathgen-clip', 'biomedclip-v2']:
    print(f"\n{'='*50}")
    print(f"Testing: {model_name}")
    print('='*50)
    try:
        # 1. 加载模型
        fm = load_fm(model_name, device=DEVICE)
        print(f"✅ load_fm OK")

        # 2. 包成 Trident encoder
        encoder = CustomInferenceEncoder(
            enc_name=fm.name,
            model=fm.model,
            transforms=fm.preprocess,
            precision=fm.precision,
        )
        print(f"✅ CustomInferenceEncoder OK")

        # 3. 跑一张图
        img_size = fm.image_size or 224
        dummy = Image.fromarray(
            (torch.rand(img_size, img_size, 3) * 255).byte().numpy()
        )
        x = fm.preprocess(dummy).unsqueeze(0).to(DEVICE)
        if fm.precision == torch.float16:
            x = x.half()
        feat = fm.encode_image(x)
        print(f"✅ encode_image OK — output shape: {feat.shape}")

        results[model_name] = "✅ PASS"

    except Exception as e:
        print(f"❌ FAILED: {e}")
        results[model_name] = f"❌ FAIL: {e}"

# 最后打印汇总
print(f"\n{'='*50}")
print("SUMMARY")
print('='*50)
for name, status in results.items():
    print(f"{name:20s} {status}")