from pathlib import Path
from speechbrain.inference.separation import SepformerSeparation as Sepformer


def get_model(path: Path):
    model = Sepformer.from_hparams(
        source="speechbrain/sepformer-wsj02mix",
        savedir=path
    )
    return model

if __name__=="__main__":
    import torchaudio
    model = get_model(path=Path('speechbrain/sepformer-wsj02mix'))
    # for custom file, change path
    est_sources = model.separate_file(
        path='speechbrain/sepformer-wsj02mix/test_mixture.wav'
    )
    torchaudio.save("source1hat.wav", est_sources[:, :, 0].detach().cpu(), 8000)
    torchaudio.save("source2hat.wav", est_sources[:, :, 1].detach().cpu(), 8000)
