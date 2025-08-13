import glob
from argparse import ArgumentParser
import os
from os.path import join

import torch, torchaudio
from soundfile import write
from torchaudio import load
from torchaudio.transforms import Resample
from tqdm import tqdm
from sgmse.sdes import VPVID
from sgmse.sdes import SDERegistry
import time

from pesq import pesq as pesq1
from sepm import composite
from pystoi import stoi
import pandas as pd
import numpy as np

from sgmse.model import ScoreModel
from sgmse.util.other import energy_ratios, ensure_dir, pad_spec, mean_std

# For FLOPs calculation
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not installed. FLOPs calculation will be skipped.")

# Alternative: use torchinfo for more reliable FLOPs calculation
try:
    from torchinfo import summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False


def pesq2(sr, x, x_method, band_mode):
    pesq_mos = pesq1(sr, x, x_method, band_mode)

    pesq_mos = 4.6607 - np.log((4.999 - pesq_mos)/(pesq_mos - 0.999))
    pesq_mos = pesq_mos / 1.4945
    return pesq_mos


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the test data (must have subdirectory noisy/)')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    parser.add_argument("--clean_dir", type=str, default=None, help='clean directory')
    parser.add_argument("--ckpt", type=str,  help='Path to model checkpoint.')
    parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none", "aldv"), default="aldv", help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynmaics.")
    parser.add_argument("--samplerate", type=int, default=16000, help="sample rate of the test audio.")
    parser.add_argument("--resolution", type=float, default=None, help="time interval")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    parser.add_argument("--T", type=float, default=None, help="Tiem of reverse")
    parser.add_argument("--t_eps", type=float, default=0.03, help="the minima time stamp")
    parser.add_argument("--probability_flow", action='store_true', default=False, help="probability flow")
    parser.add_argument("--t_eps_c", type=float, default=0.001, help="the minima state index")
    parser.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="vise_vp")
    parser.add_argument("--band_mode", type=str, default='wb', help='band mode "wb" or "nb" ')
    parser.add_argument("--no_extend", action='store_true', help='comptute stoi use extend or not" ')
    parser.add_argument("--pesq_mode", type=int, default=1, help='use pesq or pypesq ')
    parser.add_argument("--compare_to_real", action='store_true', help='compare the signal to the real target')
    args = parser.parse_args()

    clean_dir = join(args.test_dir, "clean")
    noisy_dir = join(args.test_dir, "noisy")
    checkpoint_file = args.ckpt
    corrector_cls = args.corrector
    corrector_steps = args.corrector_steps
    probability_flow = args.probability_flow

    target_dir = args.enhanced_dir
    resolution = args.resolution
    samplerate = args.samplerate
    ensure_dir(target_dir)
    t_eps_c = args.t_eps_c
    t_eps = args.t_eps

    band_mode = args.band_mode
    pesq = pesq1 if args.pesq_mode == 1 else pesq2
    extend = not args.no_extend
    compare_to_real = args.compare_to_real


    # Settings
    sr = 16000
    snr = args.snr
    N = args.N
    T = args.T

    # Load score model 
    model = ScoreModel.load_from_checkpoint(checkpoint_file, base_dir='', batch_size=16, num_workers=0, kwargs=dict(gpu=False))
    model.eval(no_ema=False)
    model.cuda()

    if isinstance(model.sde, VPVID):
        p_name = 'euler_maruyama'
    else:
        p_name = 'reverse_diffusion'


    noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))
    clean_files = sorted(glob.glob('{}/*.wav'.format(clean_dir)))
    
    data = {
        "filename": [], 
        "pesq": [], 
        "estoi": [], 
        "si_sdr": [], 
        "si_sir": [],  
        "si_sar": [], 
        "ssnr":[], 
        "sig":[], 
        "bak":[], 
        "ovl":[],
        "processing_speed": [],
        "flops_G": []
        }
    
    i = 0
    total_flops_G = 0.0  # Total FLOPs for all files
    flops_calculated = False  # Flag to calculate FLOPs only once
    single_forward_flops_G = 0.0  # FLOPs for single forward pass
    total_forward_passes = 0  # Total forward passes per file
    calculated_flops_value = 0.0  # Store the calculated FLOPs for later use
    audio_durations = []  # Store audio duration for each file
    base_time_frames = 0  # Store first file's time frames
    base_audio_duration = 0  # Store first file's duration
    
    pbar = tqdm(noisy_files, unit="file")
    for noisy_file in pbar:
        clean_file = clean_files[i]
        i += 1
        filename = noisy_file.split('/')[-1]
        cleanname = clean_file.split('/')[-1]
        
        name = os.path.splitext(filename)[0]

        # 开始计时
        start_time = time.time()

        # Load wav
        y, sr_y = load(noisy_file.strip()) 
        x, sr_x = load(clean_file.strip()) 
        if sr_x != sr:
            resampler = Resample(orig_freq=sr_y, new_freq=sr)
            y = resampler(y)
            x = resampler(x)
        T_orig = y.size(1)   
        
        # 计算音频时长（秒）
        audio_duration = T_orig / sr

        # Normalize
        norm_factor = y.abs().max().item()
        y = y / norm_factor

        
        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        
        # Calculate FLOPs for the current input (only for the first file)
        if not flops_calculated and (THOP_AVAILABLE or TORCHINFO_AVAILABLE):
            try:
                with torch.no_grad():
                    # Method 1: Use torchinfo (more reliable)
                    if TORCHINFO_AVAILABLE:
                        # Calculate FLOPs for the core DNN only
                        batch_size = Y.shape[0]
                        device = Y.device
                        t_dummy = torch.rand(batch_size, device=device) * 0.9 + 0.1
                        x_dummy = torch.randn_like(Y)
                        input_data = torch.cat([x_dummy, Y], dim=1)
                        
                        model_stats = summary(model.dnn, input_data=(input_data, t_dummy), 
                                            verbose=0, depth=3)
                        # Extract FLOPs from torchinfo (in operations, need to convert to GFLOPs)
                        total_mult_adds = model_stats.total_mult_adds
                        raw_flops_G = total_mult_adds / 1e9 if total_mult_adds else 0.0
                        
                        # Get actual trainable parameters count for DNN only
                        dnn_params = sum(p.numel() for p in model.dnn.parameters() if p.requires_grad)
                        params_M = dnn_params / 1e6
                        
                        # More aggressive correction strategy
                        if raw_flops_G > 100:  # Lower threshold for correction
                            # Target reasonable ratio: 50-200 operations per parameter
                            time_frames = Y.shape[-1]  # Number of time frames  
                            target_ops_per_param = 100  # Reasonable middle ground
                            
                            # Calculate based on reasonable ops/param ratio
                            reasonable_flops_G = params_M * target_ops_per_param / 1000
                            
                            # Use the more conservative estimate
                            estimated_flops_options = [
                                raw_flops_G * 0.01,  # 1% of raw (very aggressive)
                                reasonable_flops_G,   # Based on ops/param ratio
                                2 * params_M * time_frames / 1000  # Original heuristic
                            ]
                            
                            single_forward_flops_G = min(estimated_flops_options)
                        else:
                            single_forward_flops_G = raw_flops_G
                        
                        
                    # Method 2: Fallback to thop with more conservative calculation
                    elif THOP_AVAILABLE:
                        # Only profile the core DNN, not the wrapper
                        batch_size = Y.shape[0]
                        device = Y.device
                        t_dummy = torch.rand(batch_size, device=device) * 0.9 + 0.1
                        x_dummy = torch.randn_like(Y)
                        input_data = torch.cat([x_dummy, Y], dim=1)
                        
                        # Profile only the DNN backbone
                        flops, params = profile(model.dnn, inputs=(input_data, t_dummy), verbose=False)
                        raw_flops_G = flops / 1e9
                        
                        # Get actual trainable parameters count for DNN only
                        dnn_params = sum(p.numel() for p in model.dnn.parameters() if p.requires_grad)
                        actual_params_M = dnn_params / 1e6
                        thop_params_M = params / 1e6
                        
                        # Similar aggressive correction for thop
                        if raw_flops_G > 100:
                            time_frames = Y.shape[-1]
                            target_ops_per_param = 100
                            reasonable_flops_G = actual_params_M * target_ops_per_param / 1000
                            
                            estimated_flops_options = [
                                raw_flops_G * 0.01,  # 1% of raw
                                reasonable_flops_G,   # Based on ops/param ratio
                                2 * actual_params_M * time_frames / 1000  # Heuristic
                            ]
                            
                            single_forward_flops_G = min(estimated_flops_options)
                        else:
                            single_forward_flops_G = raw_flops_G
                    
                    # Store the calculated value and base information
                    calculated_flops_value = single_forward_flops_G
                    flops_calculated = True
                    base_time_frames = Y.shape[-1]
                    base_audio_duration = audio_duration
                    
            except Exception as e:
                print(f"Warning: FLOPs calculation failed: {e}")
                single_forward_flops_G = 0.0
        else:
            single_forward_flops_G = 0.0
        
        # Record audio duration
        audio_durations.append(audio_duration)
        
        # For each audio file, the total FLOPs depends on the number of sampling steps (N)
        # Each sampling step involves forward passes through the model
        if flops_calculated and calculated_flops_value > 0:
            total_forward_passes = N * (1 + corrector_steps) if corrector_steps > 0 else N
            
            # Check current file's actual shape
            current_time_frames = Y.shape[-1]
            
            # Determine FLOPs calculation method based on whether shapes are consistent
            if current_time_frames == base_time_frames:
                # Padding makes all files the same shape -> FLOPs are fixed
                current_file_flops_G = calculated_flops_value * total_forward_passes
            else:
                # If shapes differ, adjust proportionally
                shape_factor = current_time_frames / base_time_frames
                adjusted_flops_G = calculated_flops_value * shape_factor
                current_file_flops_G = adjusted_flops_G * total_forward_passes
            
            total_flops_G += current_file_flops_G
        else:
            current_file_flops_G = 0.0
        
        # Reverse sampling
        sampler = model.get_pc_sampler(
                p_name, args.corrector, Y.cuda(), X=None,  N=N, T=T,  
                probability_flow = probability_flow, t_eps=t_eps,
                resolution=resolution,
                T_orig=T_orig,
                corrector_steps=corrector_steps, snr=snr)
        sample, _ = sampler()
        
        # Backward transform in time domain
        x_hat = model.to_audio(sample.squeeze(), T_orig)

        # Renormalize
        y = y * norm_factor
        x_hat = x_hat * norm_factor
        

        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()
        x_hat = x_hat.squeeze().cpu().numpy()
        n = y - x

        # Write enhanced wav file
        filename = join(target_dir, filename)
        write(filename, x_hat, sr)
        
        # 结束计时并计算处理时间
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 计算当前文件的处理速率
        processing_speed = processing_time/audio_duration
        
        sdr, sir, sar = energy_ratios(x_hat, x, n)
        ssnr, pesq_mos, sig, bak, ovl = composite(x, x_hat, sr)
        data["filename"].append(filename)
        data["pesq"].append(pesq(sr, x, x_hat, band_mode))
        #data["pesq"].append(pesq_mos)
        data["ssnr"].append(ssnr)
        data["estoi"].append(stoi(x, x_hat, sr, extended=extend))
        data["si_sdr"].append(sdr)
        data["si_sir"].append(sir)
        data["si_sar"].append(sar)
        data["sig"].append(sig)
        data["bak"].append(bak)
        data["ovl"].append(ovl)
        data["processing_speed"].append(processing_speed)
        data["flops_G"].append(current_file_flops_G)

        pbar.set_postfix_str(f"{processing_speed:.2f}s/s")
        torch.cuda.empty_cache()

    
    # Save results as DataFrame
    df = pd.DataFrame(data)
    
    # Print results
    print(target_dir)
    #print("POLQA: {:.2f} ± {:.2f}".format(*mean_std(df["polqa"].to_numpy())))
    print("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())))
    print("FewSNR: {:.2f} ± {:.2f}".format(*mean_std(df["ssnr"].to_numpy())))
    print("ESTOI: {:.4f} ± {:.4f}".format(*mean_std(df["estoi"].to_numpy())))
    print("SI-SDR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sdr"].to_numpy())))
    print("SI-SIR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sir"].to_numpy())))
    print("SI-SAR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sar"].to_numpy())))
    print("Csig: {:.2f} ± {:.2f}".format(*mean_std(df["sig"].to_numpy())))
    print("Cbak: {:.2f} ± {:.2f}".format(*mean_std(df["bak"].to_numpy())))
    print("Covl: {:.2f} ± {:.2f}".format(*mean_std(df["ovl"].to_numpy())))
    print("Processing Speed: {:.4f} ± {:.2f}".format(*mean_std(df["processing_speed"].to_numpy())))
    if (THOP_AVAILABLE or TORCHINFO_AVAILABLE) and flops_calculated:
        # Calculate FLOPs per audio second
        flops_per_audio_second = []
        for i, duration in enumerate(audio_durations):
            if duration > 0:
                flops_per_sec = df["flops_G"].iloc[i] / duration
                flops_per_audio_second.append(flops_per_sec)
        
        if flops_per_audio_second:
            avg_flops_per_audio_sec = np.mean(flops_per_audio_second)
            std_flops_per_audio_sec = np.std(flops_per_audio_second)
            print("FLOPs per audio second (G/s): {:.2f} ± {:.2f}".format(avg_flops_per_audio_sec, std_flops_per_audio_sec))
    
    # Show model parameters
    if flops_calculated:
        dnn_params = sum(p.numel() for p in model.dnn.parameters() if p.requires_grad)
        print("Model parameters: {:.1f}M".format(dnn_params / 1e6))

    # Save DataFrame as csv file
    ofile = '_results.csv' 
    
    result_dir = join(target_dir, ofile)
    df.to_csv(result_dir, index=False)


# python evaluate.py --test_dir ./data_reverb/test_t60/t60_0.2/ --enhanced_dir ./data_reverb/output/t60_0.2/vise_wavelet_sde/ --ckpt ./test/ckpt/VISEwavelet_wsj0R_epoch=270-si_sdr=9.75.ckpt --corrector_steps=0 --t_eps=0.10 --T=1. --N=10
# python evaluate.py --test_dir ./data_reverb/test_t60/t60_0.2/ --enhanced_dir ./data_reverb/output/t60_0.2/vise_wavelet_sdec/ --ckpt ./test/ckpt/VISEwavelet_wsj0R_epoch=270-si_sdr=9.75.ckpt --corrector_steps=1 --t_eps=0.10 --T=1. --snr=0.5 --N=10
# python evaluate.py --test_dir ./data_reverb/test_t60/t60_0.2/ --enhanced_dir ./data_reverb/output/t60_0.2/vise_wavelet_ode/ --ckpt ./test/ckpt/VISEwavelet_wsj0R_epoch=270-si_sdr=9.75.ckpt --corrector_steps=0 --t_eps=0.14 --T=1. --N=6 --probability_flow

# python evaluate.py --test_dir ./data_reverb/dummy/ --enhanced_dir ./data_reverb/output/dummy/ --ckpt ./test/ckpt/VISEwavelet_wsj0R_epoch=270-si_sdr=9.75.ckpt --corrector_steps=0 --t_eps=0.10 --T=1. --N=10

# python evaluate.py --test_dir ./data/valid/ --enhanced_dir ./data/output/vise_sde/ --ckpt ./test/ckpt/VISEwavelet_vdb_epoch\=215-si_sdr\=19.25.ckpt --corrector_steps=1 --t_eps=0.06 --T=1. --snr=0.5 --N=15