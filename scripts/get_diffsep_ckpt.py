from huggingface_hub import hf_hub_download
from src.models.diffsep.pl_model import DiffSepModel
import argparse
from pathlib import Path

DEFAULT_MODEL = "fakufaku/diffsep"

def str_or_int(x):
    try:
        x = int(x)
    except ValueError:
        pass
    return x

def get_model(args):
    if not args.model.exists():
        # assume this is a HF model
        path = hf_hub_download(repo_id=str(args.model), filename="checkpoint.pt")
    else:
        path = args.model

    # load model
    model = DiffSepModel.load_from_checkpoint(str(path))

    # transfer to GPU
    model = model.to(args.device)
    model.eval()

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

    kwargs = {
        "N": N,
        "denoise": denoise,
        "intermediate": False,
        "corrector_steps": corrector_steps,
        "snr": snr,
        "schedule": args.schedule,
    }

    return model, kwargs

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Separate all the wav files in a specified folder"
    )
    parser.add_argument("input_dir", type=Path, help="Path to the input folder")
    parser.add_argument("output_dir", type=Path, help="Path to the output folder")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to model or Huggingface model",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str_or_int,
        default="cuda:0",
        help="Device to use (default: cuda:0)",
    )
    parser.add_argument("-N", type=int, default=None, help="Number of steps")
    parser.add_argument(
        "--snr", type=float, default=None, help="Step size of corrector"
    )
    parser.add_argument(
        "--corrector-steps", type=int, default=None, help="Number of corrector steps"
    )
    parser.add_argument(
        "--denoise", type=bool, default=True, help="Use denoising in solver"
    )
    parser.add_argument(
        "-s", "--schedule", type=str, help="Pick a different schedule for the inference"
    )
    args = parser.parse_args()
    model,kwargs = get_model(args=args)
    print(model)