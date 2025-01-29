import librosa
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from src.models.facodec import FACodecEncoder,FACodecDecoder


def load_audio(wav_path):
    wav = librosa.load(wav_path, sr=16000)[0]
    wav = torch.from_numpy(wav).float()
    wav = wav.unsqueeze(0).unsqueeze(0)
    return wav


fa_encoder = FACodecEncoder(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
)

fa_decoder = FACodecDecoder(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=True,
    use_gr_residual_f0=True,
    use_gr_residual_phone=True,
)

encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")

fa_encoder.load_state_dict(torch.load(encoder_ckpt))
fa_decoder.load_state_dict(torch.load(decoder_ckpt))

fa_encoder.eval()
fa_decoder.eval()

with torch.no_grad():
    path = '/data/milsrg1/corpora/LibriMix/Libri2Mix/wav16k/max/train-100/mix_both/6000-55211-0003_322-124146-0008.wav'
    test_wav = load_audio(path)
    # encode
    enc_out = fa_encoder(test_wav)
    print("Enc out shape:\n")
    print(enc_out.shape)
    # quantize
    vq_post_emb, vq_id, _, quantized, spk_embs = fa_decoder(enc_out, eval_vq=False, vq=True)
    # latent after quantization
    print(vq_post_emb.shape)
    # codes
    print("vq id shape:", vq_id.shape)
    # get prosody code
    prosody_code = vq_id[:1]
    print("prosody code shape:", prosody_code.shape)
    # print("prosody code:", prosody_code)
    # get content code
    cotent_code = vq_id[1:3]
    print("content code shape:", cotent_code.shape)
    # get residual code (acoustic detail codes)
    residual_code = vq_id[3:]
    print("residual code shape:", residual_code.shape)
    # print("residual code:", residual_code)
    # speaker embedding
    print("speaker embedding shape:", spk_embs.shape)
    # decode (recommand)
    recon_wav = fa_decoder.inference(vq_post_emb, spk_embs)
    print(recon_wav.shape)
