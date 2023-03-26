import argparse
from contextlib import nullcontext
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from imwatermark import WatermarkEncoder
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from tqdm import tqdm

torch.set_grad_enabled(False)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir", type=str, nargs="?", help="dir to write results to", default="outputs/images"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action="store_true",
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm",
        action="store_true",
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=768,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=768,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="full",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device on which Stable Diffusion will be run",
        choices=["cpu", "cuda"],
        default="cuda",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="../input/metadata.parquet",
    )
    parser.add_argument(
        "--part_id",
        type=int,
        default=1,
        required=True,
    )
    parser.add_argument(
        "--H_dest",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--W_dest",
        type=int,
        default=512,
    )
    opt = parser.parse_args()
    return opt


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, "dwtDct")
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_metadata(metadata_path, part_id):
    df = pd.read_parquet(metadata_path)
    print(f"metadata.shape={df.shape}")

    print("drop duplicated prompts...")
    df = df.drop_duplicates(subset="prompt", keep="first", ignore_index=True)
    print(f"metadata.shape={df.shape}")

    print(f"part_id={part_id}")
    df = df.query(f"part_id == {part_id}").reset_index(drop=True)
    print(f"metadata.shape={df.shape}")

    return df


def main(opt):
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")

    metadata = load_metadata(opt.metadata_path, opt.part_id)

    outdir = Path(opt.outdir) / f"part-{opt.part_id:06d}"
    outdir.mkdir(exist_ok=True, parents=True)

    exist_pngs = set([path.name for path in outdir.glob("*.png")])
    metadata = metadata[~metadata.image_name.isin(exist_pngs)].reset_index(drop=True)
    if len(metadata) == 0:
        print("all pngs are already generated.")
        return
    print(f"metadata.shape={metadata.shape}")

    device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
    model = load_model_from_config(config, f"{opt.ckpt}", device)

    if opt.plms:
        sampler = PLMSSampler(model, device=device)
    elif opt.dpm:
        sampler = DPMSolverSampler(model, device=device)
    else:
        sampler = DDIMSampler(model, device=device)

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark("bytes", wm.encode("utf-8"))

    data = list(chunk(metadata.prompt.to_list(), opt.n_samples))
    all_names = list(chunk(metadata.image_name.to_list(), opt.n_samples))

    start_code = None

    precision_scope = autocast if opt.precision == "autocast" or opt.bf16 else nullcontext
    with torch.inference_mode(), precision_scope(opt.device), model.ema_scope():
        for prompts, names in zip(tqdm(data, desc="data"), all_names):
            batch_size = min(len(prompts), opt.n_samples)
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(batch_size * [""])
            if isinstance(prompts, tuple):
                prompts = list(prompts)
            c = model.get_learned_conditioning(prompts)
            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
            samples, _ = sampler.sample(
                S=opt.steps,
                conditioning=c,
                batch_size=batch_size,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=opt.scale,
                unconditional_conditioning=uc,
                eta=opt.ddim_eta,
                x_T=start_code,
            )

            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

            for x_sample, name in zip(x_samples, names):
                x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
                x_sample = cv2.resize(x_sample, (opt.W_dest, opt.H_dest), interpolation=cv2.INTER_NEAREST)
                img = Image.fromarray(x_sample.astype(np.uint8))
                img = put_watermark(img, wm_encoder)
                img.save(outdir / name)


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
