import concurrent.futures
import subprocess

import pandas as pd

import torch


DIV = 3
MOD = 1


def run(gpu_id, part_id):
    cmd = " ".join([
        f"CUDA_VISIBLE_DEVICE={gpu_id}",
        "python",
        "scripts/txt2img_db.py",
        "--ckpt",
        "stable-diffusion-2/768-v-ema.ckpt",
        "--config",
        "configs/stable-diffusion/v2-inference-v.yaml",
        "--H",
        "768",
        "--W",
        "768",
        "--precision",
        "full",
        "--device",
        "cuda",
        "--outdir",
        "./outputs/images",
        "--metadata_path",
        "../input/metadata.parquet",
        "--part_id",
        f"{part_id}",
        "--H_dest",
        "512",
        "--W_dest",
        "512",
        "--n_samples",
        "6"
    ])
    print(cmd)
    subprocess.run(cmd, shell=True)


def main():
    if not torch.cuda.is_available():
        return

    device_count = torch.cuda.device_count()

    metadata = pd.read_parquet("../input/metadata.parquet")
    part_ids = sorted([pid for pid in metadata.part_id.unique() if pid % DIV == 1])
    print(f"len(part_ids)={len(part_ids)}")

    while len(part_ids) > 0:
        with concurrent.futures.ProcessPoolExecutor(max_workers=device_count) as executor:
            futures = []
            for gpu_id in range(device_count):
                curr_pid = part_ids.pop(0)
                future = executor.submit(run, gpu_id, curr_pid)
                futures.append(future)
            _ = [future.result() for future in futures]


if __name__ == "__main__":
    main()
