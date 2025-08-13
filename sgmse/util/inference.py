import torch
from torchaudio import load
from torchaudio.transforms import Resample

from pesq import pesq
from pystoi import stoi

from .other import si_sdr, pad_spec
from ..sdes import VPVID

# Settings
sr = 16000
snr = 0.5
N = None
corrector_steps = 0


def evaluate_model(model, num_eval_files, p_name='reverse_diffusion',
                   probability_flow=False, corrector='none', snr=0.5):
    if corrector == 'none':
        corrector_steps = 0
    else:
        corrector_steps = 1
        # print(f"Using corrector: {corrector}, snr: {snr}, corrector_steps: {corrector_steps}")

    clean_files = model.data_module.valid_set.clean_files
    noisy_files = model.data_module.valid_set.noisy_files
    
    # Select test files uniformly accros validation files
    total_num_files = len(clean_files)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    clean_files = list(clean_files[i] for i in indices)
    noisy_files = list(noisy_files[i] for i in indices)

    _pesq = 0
    _si_sdr = 0
    _estoi = 0

    # print(f'p_name: {p_name}, probability_flow: {probability_flow}')
    
    # iterate over files
    for (clean_file, noisy_file) in zip(clean_files, noisy_files):
        # Load wav
        y, sr_y = load(noisy_file.strip()) 
        x, sr_x = load(clean_file.strip()) 
        if sr_x != sr:
            resampler = Resample(orig_freq=sr_y, new_freq=sr)
            y = resampler(y)
            x = resampler(x)
        T_orig = x.size(1)   

        # Normalize per utterance
        norm_factor = y.abs().max()
        y = y / norm_factor

        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        y = y * norm_factor

        # Reverse sampling
        sampler = model.get_pc_sampler(
            p_name, corrector, Y.cuda(), N=N, 
            probability_flow=probability_flow, corrector=corrector,
            corrector_steps=corrector_steps, snr=snr)
        sample, _ = sampler()

        x_hat = model.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor

        x_hat = x_hat.squeeze().cpu().numpy()
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()

        # print("x.shape:", x.shape)
        # print("x_hat.shape:", x_hat.shape)

        _si_sdr += si_sdr(x, x_hat)
        _pesq += pesq(sr, x, x_hat, 'wb') 
        _estoi += stoi(x, x_hat, sr, extended=True)
        
    return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files

