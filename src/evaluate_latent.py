# 2023 (c) LINE Corporation
# Authors: Robin Scheibler
# MIT License
import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import fast_bss_eval
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import yaml
import logging
import hydra
from omegaconf import OmegaConf, DictConfig
from pesq import pesq
from pystoi import stoi

import utils
from datasets import WSJ0_mix
from diffsep_latent import LatentDiffSep

os.environ["HYDRA_FULL_ERROR"] = "1"


def get_default_datasets(n_spkr=2, fs=8000, USE_WSJ0 = False, USE_LIBRIMIX=True):
    ds = OmegaConf.load("src/config/diffsep/datamodule/default.yaml")
    if USE_WSJ0:
        for split in ["val", "test", "train"]:
            ds[split].dataset.path = "/data/milsrg1/corpora/wsj0_mix"
    if USE_LIBRIMIX:
        for split in ["librimix_test", "librimix_train-100", "librimix_train-360"]:
            ds[split].dataset.path = "/data/milsrg1/corpora/LibriMix"
    for split in ds:
        if "_target_" in ds[split].dataset:
            ds[split].dataset.pop("_target_")
        ds[split].dataset.n_spkr = n_spkr
        ds[split].dataset.fs = fs
    return ds


def save_fig(x_result, intmet, target, fig_out_fn, n_fig=6, vmin=-75, vmax=0):
    # back to cpu
    x_result = x_result.cpu()
    target = target.cpu()

    # Save figure of evolution
    fig, axes = plt.subplots(2, n_fig + 1, figsize=(20, 4))

    steps = np.round(np.linspace(0, 1, n_fig) * (len(intmet) - 1)).astype(np.int64)

    for idx, step in enumerate(steps):
        arr = intmet[step][0].cpu().numpy()
        for i in range(2):
            im = axes[i, idx].specgram(arr[0, i], vmin=vmin, vmax=vmax)
            axes[i, idx].set_xticks([])
            axes[i, idx].set_yticks([])
            if i == 0:
                axes[i, idx].set_title(
                    f"t={(len(intmet) - 1 - step) / (len(intmet) - 1):.2f}"
                )
    for i in range(2):
        tgt = target[0, i] + np.random.randn(*target[0, i].shape) * 1e-10
        *_, im = axes[i, -1].specgram(tgt, vmin=vmin, vmax=vmax)
        axes[i, -1].set_xticks([])
        axes[i, -1].set_yticks([])
        if i == 0:
            axes[i, -1].set_title("clean")
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.savefig(fig_out_fn)
    plt.close(fig)


def save_samples(mix, x_result, target, wav_out_fn, fs):
    # save samples
    all_wav = torch.cat((mix[0].cpu(), x_result[0, :].cpu(), target[0].cpu()), dim=0,)
    all_wav = all_wav.cpu()
    all_wav = all_wav[:, None, :]

    max_val = abs(all_wav).max()
    all_wav *= 0.95 / max_val

    """
    torchaudio.save(
        str(wav_out_fn.with_suffix(".mix.wav")),
        all_wav[0],
        fs,
    )
    """
    torchaudio.save(
        str(wav_out_fn.with_suffix(".enh0.wav")), all_wav[1], fs,
    )
    torchaudio.save(
        str(wav_out_fn.with_suffix(".enh1.wav")), all_wav[2], fs,
    )

    torchaudio.save(
        str(wav_out_fn.with_suffix(".tgt0.wav")),
        all_wav[3],
        fs,
    )
    torchaudio.save(
        str(wav_out_fn.with_suffix(".tgt1.wav")),
        all_wav[4],
        fs,
    )



def compute_metrics(ref, est, fs, pesq_mode="nb", stoi_extended=True):

    si_sdr, si_sir, si_sar, perm = fast_bss_eval.si_bss_eval_sources(
        ref, est, zero_mean=False, compute_permutation=True, clamp_db=100,
    )

    # order according to SIR
    est = est[:, perm[0], :]

    est = est.cpu().numpy()
    ref = ref.cpu().numpy()

    p_esq = []
    s_toi = []
    for src_idx in range(est.shape[-2]):
        p_esq.append(pesq(fs, ref[0, src_idx], est[0, src_idx], pesq_mode))
        s_toi.append(stoi(ref[0, src_idx], est[0, src_idx], fs, extended=stoi_extended))

    return si_sdr, si_sir, si_sar, p_esq, s_toi, perm


def summarize(results, ignore_inf=True):
    metrics = set()
    summary = defaultdict(lambda: 0)
    denominator = defaultdict(lambda: 0)

    for res in results.values():
        for met, val in res.items():
            metrics.add(met)
            val_mean = np.mean(val)
            if ignore_inf or not np.isinf(val_mean):
                summary[met] += val_mean
                denominator[met] += 1
        summary["number"] += 1

    for met in metrics:
        summary[met] = (summary[met] / denominator[met]).tolist()

    return dict(summary)


def evaluate_process(args, output_dir, split, start_idx, stop_idx, device, model_config:DictConfig):

    fig_dir = output_dir / "fig"
    wav_dir = output_dir / "wav"

    if args.dl_workers is None:
        num_dl_workers = os.cpu_count()
    else:
        num_dl_workers = args.dl_workers

    # special case to get the original data
    no_proc_flag = str(args.ckpt) == "__no_proc__"

    default_datasets = get_default_datasets()

    if no_proc_flag:
        # load validation dataset
        dataset = WSJ0_mix(path="data/wsj0_mix", n_spkr=2, cut="max", split=split)

    else:

        if split in model_config.datamodule:
            ds_args = model_config.datamodule[split].dataset
        else:
            ds_args = default_datasets[split].dataset

        # remove the target because we don't use 'instantiate'
        if "_target_" in ds_args:
            del ds_args._target_
        # ds_args.pop("_target_", None)
        # check the location of the data
        data_path = Path(ds_args.path)
        if not data_path.exists():
            if split in ["val", "test"]:
                ds_args.path = "./data/wsj0_mix"
            else:
                ds_args.path = "./data/LibriMix"

        # load validation dataset
        dataset = WSJ0_mix(**ds_args)

        #transform the config to a DictConfig
        #model = DiffSepModel.load_from_checkpoint(str(args.ckpt), config = model_config)
        model =  LatentDiffSep(model_config)
        ckpt = torch.load(str(args.ckpt))["state_dict"]
        model.load_state_dict(ckpt, strict=True)

        # transfer to GPU
        model = model.to(device)
        model.eval(no_ema=True)

        # prepare inference parameters
        sampler_kwargs = model.config.model.sampler
        N = sampler_kwargs.N if args.N is None else args.N
        corrector_steps = (
            sampler_kwargs.corrector_steps
            if args.corrector_steps is None
            else args.corrector_steps
        )
        snr = sampler_kwargs.snr if args.snr is None else args.snr
        denoise = args.denoise

    # save the sampling freq of the dataset
    fs = dataset.fs

    # we only access a subset of the dataset
    dataset = torch.utils.data.Subset(dataset, range(start_idx, stop_idx))

    # wraps datasets into dataloaders
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        num_workers=num_dl_workers,
        pin_memory=True,
        batch_size=1,
    )

    results = dict()

    for batch_idx, (mix, target) in zip(range(start_idx, stop_idx), dataloader):
        print(f"Initial shape of mix: {mix.shape}, of target: {target.shape}")
        # decide if we want to save some sample and figure
        save_samples_fig = args.save_n is None or (batch_idx < args.save_n)

        #mix = mix.to(device)
        mix_latent = mix.clone().to(device)
        target_latent = target.clone().to(device)
        target = target.to(device)
        length = target.shape[-1] / fs

        if no_proc_flag:
            x_result = torch.broadcast_to(mix, target.shape)
            nfe = 0
            intmet = None
            t_proc = 0.0
            save_samples_fig = False

        else:

            mix_latent, target_latent = model.encode(mix_latent, target_latent)
            print(f"""Initial shape of mix_latent: {mix_latent.shape}, of target_latent: {target_latent.shape}""")

            sampler = model.get_pc_sampler(
                "reverse_diffusion",
                "ald",
                mix_latent,
                N=N,
                denoise=denoise,
                intermediate=save_samples_fig,
                corrector_steps=corrector_steps,
                snr=snr,
                schedule=args.schedule,
            )

            t_s = time.perf_counter()
            x_result, nfe, *others = sampler()
            x_result = model.decode(x_result, target.shape[-1])

            t_proc = time.perf_counter() - t_s

            if len(others) > 0:
                intmet = others[0]

        # compute the metrics before separation
        si_sdr, si_sir, si_sar, p_esq, s_toi, perm = compute_metrics(
            target,
            x_result,
            fs,
            pesq_mode=args.pesq_mode,
            stoi_extended=not args.stoi_no_extended,
        )

        # fix the permutation
        x_result = x_result[:, perm[0], :]

        results[batch_idx] = {
            "batch_idx": batch_idx,
            "si_sdr": si_sdr.tolist(),
            "si_sir": si_sir.tolist(),
            "si_sar": si_sar.tolist(),
            "pesq": p_esq,
            "stoi": s_toi,
            "nfe": nfe,
            "runtime": t_proc,
            "len_s": length,
        }

        if start_idx == 0:
            # only print for a single process
            print(f"{split}", end=" ")
            for met, val in results[batch_idx].items():
                print(f"{met}={np.mean(val):.3f}", end=" ")
            print()

        if save_samples_fig:

            # fix permutations of intermediate results
            if intmet is not None:
                for idx in range(len(intmet)):
                    xt, xt_mean = intmet[idx]
                    intmet[idx] = (xt[:, perm[0], :], xt_mean[:, perm[0], :])

            fig_out_dir = fig_dir / split
            fig_out_dir.mkdir(exist_ok=True, parents=True)
            wav_out_dir = wav_dir / split
            wav_out_dir.mkdir(exist_ok=True, parents=True)

            save_fig(
                x_result,
                intmet,
                target,
                fig_out_dir / f"evo_{batch_idx:04d}.pdf",
                n_fig=6,
                vmin=-75,
                vmax=0,
            )
            print(f"Tensor shapes before save_samples: {mix.shape}, {x_result.shape}, {target.shape}")
            save_samples(mix, x_result, target, wav_out_dir / f"{batch_idx:04d}", fs)

    return split, results


def str_or_int(x):
    try:
        x = int(x)
    except ValueError:
        pass
    return x

@hydra.main(config_path="./config/latent_diffsep_ouve", config_name="config", version_base=None)
def main(model_config:DictConfig):
    torch.multiprocessing.set_start_method("spawn")

    args = dict()
    args["ckpt"] = Path("diffsep_latent/zysma1be/checkpoints/epoch-029_si_sdr--17.138.ckpt")
    args["output_dir"] = Path("results_latent")
    args["device"] = [0]
    args["workers"] = 0
    args["dl_workers"] = 0
    args["tag"] = None
    args["limit"] = None
    args["save_n"] = None
    args["N"] = None
    args["snr"] = None
    args["corrector_steps"] = None
    args["denoise"] = True
    args["pesq_mode"] = "nb"
    args["stoi_no_extended"] = False
    args["schedule"] = None
    args["n_proc"] = None
    args = OmegaConf.create(args)
    splits = ["librimix_test"]

    output_dir_base = args.output_dir

    # special case to get the original data
    no_proc_flag = str(args.ckpt) == "__no_proc__"

    if no_proc_flag:
        if args.tag is None:
            output_dir = output_dir_base / "mix"
        else:
            output_dir = output_dir_base / args.tag

    else:
        # load the config file
        #hparams = OmegaConf.load("/research/milsrg1/user_workspace/efb48/DiTSep/checkpoints/diffsep/hparams.yaml")
        #config = hparams.config

        # prepare inference parameters
        sampler_kwargs = model_config.model.sampler
        N = sampler_kwargs.N if args.N is None else args.N
        corrector_steps = (
            sampler_kwargs.corrector_steps
            if args.corrector_steps is None
            else args.corrector_steps
        )
        snr = sampler_kwargs.snr if args.snr is None else args.snr
        denoise = args.denoise
        tag_inf = f"N-{N}_snr-{snr}_corrstep-{corrector_steps}_denoise-{denoise}_schedule-{args.schedule}"

        # create folder name based on experiment and checkpoint
        exp_name = args.ckpt.parent.name
        ckpt_name = args.ckpt.stem
        if args.tag is None:
            output_dir = output_dir_base / f"{exp_name}_{ckpt_name}_{tag_inf}"
        else:
            output_dir = output_dir_base / f"{args.tag}_{tag_inf}"

    output_dir.mkdir(exist_ok=True, parents=True)
    fig_dir = output_dir / "fig"
    wav_dir = output_dir / "wav"
    print(f"Created output folder {output_dir}")

    # load default dataset config
    default_datasets = get_default_datasets()

    # now divide the tasks betwen the workers
    tasks = []
    dev_idx = 0
    for split in splits:

        # load dataset to get the length
        if no_proc_flag:
            # load validation dataset
            dataset = WSJ0_mix(path="/data/milsrg1/corpora/wsj0_mix", n_spkr=2, cut="max", split=split)
        else:
            if split in model_config.datamodule:
                ds_args = model_config.datamodule[split].dataset
            else:
                ds_args = default_datasets[split].dataset

            # remove the target because we don't use 'instantiate'
            #ds_args.pop("_target_", None)
            del ds_args._target_
            # check the location of the data
            data_path = Path(ds_args.path)
            if not data_path.exists():
                ds_args.path = "/data/milsrg1/corpora/wsj0_mix"

            # load validation dataset
            dataset = WSJ0_mix(**ds_args)

        # compute the number of files
        if args.limit is not None:
            n_samples = min(args.limit, len(dataset))
        else:
            n_samples = len(dataset)
        n_per_worker = math.floor(n_samples / max(1, args.workers))
        start_idx = 0
        while start_idx < n_samples:
            stop_idx = min(start_idx + n_per_worker, n_samples)
            tasks.append(
                (args, output_dir, split, start_idx, stop_idx, args.device[dev_idx])
            )
            start_idx = stop_idx
            dev_idx = (dev_idx + 1) % len(args.device)

    if args.workers == 0:
        results = []
        for task_args in tasks:
            results.append(evaluate_process(*task_args, model_config=model_config))
    else:
        with utils.SyncProcessingPool(args.workers) as pool:
            for task_args in tasks:
                pool.push(evaluate_process, task_args, model_config=model_config)
            results, *_ = pool.wait_results(progress_bar=True)

    # aggregate results
    agg_results = dict(zip(splits, [dict() for s in splits]))
    for (sp, res) in results:
        agg_results[sp].update(res)

    for split, results in agg_results.items():
        with open(output_dir / f"{split}.json", "w") as f:
            json.dump(results, f, indent=2)

        summary = summarize(results)
        with open(output_dir / f"{split}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary for {split}")
        print(summary)

if __name__ == "__main__":
    main()
