import logging
import os
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
import torch
import torchaudio
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig

torchaudio.set_audio_backend("sox_io")

log = logging.getLogger(__name__)

split_map = {    
    "libri2mix_train": "train-360",
    "libri2mix_test": "test",
    "libri3mix_train": "train-360",
    "libri3mix_test": "test",
}


class WSJ0_mix(torch.utils.data.Dataset):
    def __init__(
        self,
        path: Union[str, Path],
        n_spkr: Optional[int] = 2,
        fs: Optional[int] = 8000,
        cut: Optional[str] = "max",
        split: Optional[str] = "train",
        max_len_s: Optional[float] = None,
        max_n_samples: Optional[int] = None,
        mix_dir: Optional[str] = "mix",
    ):
        super().__init__()

        self.base_folder = Path(to_absolute_path(str(path)))
        self.n_spkr = n_spkr
        self.fs = int(fs)
        self.cut = cut
        self.max_len = int(self.fs * max_len_s) if max_len_s is not None else None

        if fs not in [8000, 16000]:
            raise ValueError(
                f"The sampling frequency fs can be only 8000 or 16000 (passed {fs})"
            )

        if n_spkr not in [1, 2, 3]:
            raise ValueError(
                f"The number of speakers can only be 2 or 3 (passed {n_spkr})"
            )

        if cut not in ["min", "max"]:
            raise ValueError(
                f"The cut parameter has to be 'min' or 'max' (passed {cut})"
            )

        if split not in split_map:
            raise ValueError(
                f"The split parameter must be 'train', 'val', or 'test' (passed {split})"
            )
        if "libri" in self.base_folder.name.lower():
            self.path = (
                self.base_folder
                / f"Libri{max(self.n_spkr, 2)}Mix/wav{self.fs // 1000}k/{cut}/{split_map[split]}"
            )
        else:
            self.path = (
                self.base_folder
                / f"{self.n_spkr}speakers/wav{self.fs // 1000}k/{cut}/{split_map[split]}"
            )

        self.path_mix = self.path / mix_dir
        self.path_src = [self.path / f"s{idx + 1}" for idx in range(max(self.n_spkr,2))]
        self.file_list = sorted(os.listdir(self.path_mix))

        if max_n_samples is not None:
            self.file_list = self.file_list[:max_n_samples]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        mix, *_ = torchaudio.load(self.path_mix / filename)
        tgt = torch.cat(
            [torchaudio.load(path / filename)[0] for path in self.path_src], dim=0
        )

        if self.max_len is not None and tgt.shape[-1] > self.max_len:
            # take a random cut of the right size
            p = int(torch.randint(0, tgt.shape[-1] - self.max_len, size=(1,)))
            tgt = tgt[..., p : p + self.max_len]
            mix = mix[..., p : p + self.max_len]

        return mix, tgt


class WSJ0_mix_vae(torch.utils.data.Dataset):
    def __init__(
        self,
        path: Union[str, Path],
        n_spkr: Optional[int] = 2,
        fs: Optional[int] = 8000,
        cut: Optional[str] = "max",
        split: Optional[str] = "train",
        max_len_s: Optional[float] = None,
        max_n_samples: Optional[int] = None,
        mix_dir: Optional[str] = "mix_both",
    ):
        super().__init__()

        self.base_folder = Path(to_absolute_path(str(path)))
        self.n_spkr = n_spkr
        self.fs = int(fs)
        self.cut = cut
        self.max_len = int(self.fs * max_len_s) if max_len_s is not None else None

        if fs not in [8000, 16000]:
            raise ValueError(
                f"The sampling frequency fs can be only 8000 or 16000 (passed {fs})"
            )

        if n_spkr not in [1, 2, 3]:
            raise ValueError(
                f"The number of speakers can only be 2 or 3 (passed {n_spkr})"
            )

        if cut not in ["min", "max"]:
            raise ValueError(
                f"The cut parameter has to be 'min' or 'max' (passed {cut})"
            )

        if split not in split_map:
            raise ValueError(
                f"The split parameter must be 'train', 'val', or 'test' (passed {split})"
            )
        if "libri" in self.base_folder.name.lower():
            self.path = (
                self.base_folder
                / f"Libri{max(self.n_spkr, 2)}Mix/wav{self.fs // 1000}k/{cut}/{split_map[split]}"
            )
        else:
            self.path = (
                self.base_folder
                / f"{self.n_spkr}speakers/wav{self.fs // 1000}k/{cut}/{split_map[split]}"
            )

        self.path_mix = self.path / mix_dir
        self.path_mix_clean = self.path / "mix_clean"
        self.path_src = [self.path / f"s{idx + 1}" for idx in range(max(self.n_spkr,2))]
        self.file_list = sorted(os.listdir(self.path_mix))
        self.file_list_clean = sorted(os.listdir(self.path_mix_clean))

        if max_n_samples is not None:
            self.file_list = self.file_list[:max_n_samples]
            self.file_list_clean = self.file_list_clean[:max_n_samples]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        mix, *_ = torchaudio.load(self.path_mix / filename)
        tgt = torch.cat(
            [torchaudio.load(path / filename)[0] for path in self.path_src], dim=0
        )
        mix_clean = torchaudio.load(self.path_mix_clean / filename)[0]

        if self.max_len is not None and tgt.shape[-1] > self.max_len:
            # take a random cut of the right size
            p = int(torch.randint(0, tgt.shape[-1] - self.max_len, size=(1,)))
            tgt = tgt[..., p : p + self.max_len]
            mix = mix[..., p : p + self.max_len]
            mix_clean = mix_clean[..., p : p + self.max_len]

        return mix, mix_clean, tgt

def max_collator(batch):
    """
    Collate a bunch of multichannel signals based
    on the size of the longest sample. The samples are cut at the center
    """
    max_len = max([s[0].shape[-1] for s in batch])

    new_batch = []
    for bidx, row in enumerate(batch):
        new_row = []
        for eidx, el in enumerate(row):
            if isinstance(el, torch.Tensor):
                off = max_len - el.shape[-1]
                new_row.append(torch.nn.functional.pad(el, (off // 2, off - off // 2)))
        new_batch.append(tuple(new_row))

    return torch.utils.data.default_collate(new_batch)


class WSJ0_mix_Module(pl.LightningDataModule):
    def __init__(self, dataset_name:str, config: DictConfig):
        super().__init__()
        self.dataset_name = dataset_name
        self.cfg = config
        self.datasets = {}

    def setup(self, *args, **kwargs):
        for split in self.cfg:
            self.datasets[split] = WSJ0_mix_vae(**self.cfg[split]["dataset"]) ##TODO: Change to WSJ0_mix!!!!!!!!!!!!!!!!!!!!!

    def _get_dataloader(self, split):
        return torch.utils.data.DataLoader(
            self.datasets[split],
            collate_fn=max_collator,
            **self.cfg[split]["dl_opts"],
        )

    def train_dataloader(self):
        return self._get_dataloader("train_"+ self.dataset_name)

    def val_dataloader(self):
        return self._get_dataloader("val_" + self.dataset_name)

    def test_dataloader(self):
        return self._get_dataloader("test_" + self.dataset_name)
