# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import os
import shutil
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

def main(args):
    # Setup PyTorch:
    torch.set_grad_enabled(False)
    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = '/home2/jwheo/jmpark/DiT/pretrained_models/DiT-XL-2-256x256.pt'
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    n = 500
    class_labels = []
    for f in range(int(1000 / n)):
        class_labels_tmp = np.arange(f * n , (f + 1) * n)
        class_labels.append(class_labels_tmp)
    
    # for epoch in range(0, 300): # gpu 0번 코드
    # for epoch in range(300, 600): # gpu 1번 코드
    # for epoch in range(600, 900): # gpu 2번 코드
    for epoch in range(940, 1200): # gpu 3번 코드
        # Change seed
        np.random.seed(epoch)
        torch.manual_seed(epoch)

        # Generate new class_labels
        for f in range(int(1000 / n)):
            np.random.shuffle(class_labels[f])

        for k in range(int(1000 / n)):
            print(f'starting epoch : {epoch}-{k}')
            # Create sampling noise:
            z = torch.randn(n, 4, latent_size, latent_size, device=device)
            # Save sampling noise
            print(f'-------saving noise_{epoch}-{k}-------')
            for j in range(z.shape[0]):
                np.save(os.path.join('/home2/jwheo/jmpark/DiT/new_dataset/noise(latent)', f'noise_{epoch}_{k * n + j}.npy'), z[j].cpu().numpy())
            print(f'-------done saving noise-------')
            y = torch.tensor(class_labels[k], device=device)
            # Save class_labels
            print(f'-------saving labels_{epoch}-{k}-------')
            for j in range(n):
                np.save(os.path.join('/home2/jwheo/jmpark/DiT/new_dataset/labels(latent)', f'label_{epoch}_{k * n + j}.npy'), class_labels[k][j])
            print(f'-------done saving labels-------')

            # Setup classifier-free guidance:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            
            # Sample images:
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            # samples = vae.decode(samples / 0.18215).sample

            # Save and display images:
            print(f'-------saving images_{epoch}-------')
            for j, image in enumerate(samples):
                # 이미지를 저장
                # save_image(image, file_path, normalize=False, value_range=(-1, 1))
                np.save(os.path.join('/home2/jwheo/jmpark/DiT/new_dataset/images(latent)', f'image_{epoch}_{k * n + j}.npy'), image.cpu())
                print(f'image_{epoch}_{k * n + j}')
            print(f'-------done saving images-------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.50)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
