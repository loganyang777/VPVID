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

from sgmse.model import ScoreModel
from sgmse.util.other import ensure_dir, pad_spec

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the test data (must have subdirectory noisy/)')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    parser.add_argument("--ckpt", type=str,  help='Path to model checkpoint.')
    parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none", "aldv"), default="aldv", help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=0, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynmaics.")
    parser.add_argument("--samplerate", type=int, default=16000, help="sample rate of the test audio.")
    parser.add_argument("--resolution", type=float, default=None, help="time interval")
    parser.add_argument("--N", type=int, default=10, help="Number of reverse steps")
    parser.add_argument("--T", type=float, default=0.99, help="Tiem of reverse")
    parser.add_argument("--t_eps", type=float, default=0.10, help="the minima time stamp")
    parser.add_argument("--probability_flow", action='store_true', default=False, help="probability flow")
    parser.add_argument("--t_eps_c", type=float, default=0.001, help="the minima state index")
    parser.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="vpvid")
    args = parser.parse_args()

    noisy_dir = join(args.test_dir, 'noisy/')
    if not os.path.exists(noisy_dir):
        noisy_dir = args.test_dir
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
    
    # 添加用于计算平均处理时间的变量
    total_processing_time = 0
    total_audio_duration = 0
    # 存储每个文件的处理速率
    processing_speeds = []
    
    i = 0
    pbar = tqdm(noisy_files, unit="file")
    for noisy_file in pbar:
        i += 1
        filename = noisy_file.split('/')[-1]
        
        name = os.path.splitext(filename)[0]

        # 开始计时
        start_time = time.time()

        # Load wav
        y, sr_y = load(noisy_file.strip()) 
        if sr_y != sr:
            resampler = Resample(orig_freq=sr_y, new_freq=sr)
            y = resampler(y)
        T_orig = y.size(1)   
        
        # 计算音频时长（秒）
        audio_duration = T_orig / sr

        # Normalize
        norm_factor = y.abs().max()
        y = y / norm_factor
        
        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
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
        x_hat = x_hat * norm_factor

        # Write enhanced wav file
        filename = join(target_dir, filename)
        write(filename, x_hat.cpu().numpy(), sr)
        
        # 结束计时并计算处理时间
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 计算当前文件的处理速率
        processing_speed = processing_time/audio_duration
        
        # 累计总处理时间和音频时长（保留原来的计算方式）
        total_processing_time += processing_time
        total_audio_duration += audio_duration
        
        # 添加当前文件的处理速率到列表
        processing_speeds.append(processing_speed)
        
        pbar.set_postfix_str(f"{processing_speed:.2f}s/s")
        
        torch.cuda.empty_cache()
