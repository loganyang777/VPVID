import inspect
from sgmse.backbones import BackboneRegistry
import time
from math import ceil
import warnings

import torch
import numpy as np
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage

from sgmse import sampling
from sgmse.sdes import SDERegistry, VPVID
from sgmse.mlp import MLP
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.other import pad_spec
from pesq import pesq

import os, random

from sgmse.sampling.predictors import PredictorRegistry
from sgmse.sampling.correctors import CorrectorRegistry

class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=4e-2, help="The minimum time (1e-3 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="mse", choices=("mse", "mae"), help="The type of loss function to use.")
        parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
        parser.add_argument("--time_emb", type=bool, default=True, help="using time step as condition or sigma")
        parser.add_argument("--p_name", type=str, default='euler_maruyama', choices=PredictorRegistry.get_all_names(), help=" ")        
        # parser.add_argument("--p_name", type=str, default='reverse_diffusion', choices=PredictorRegistry.get_all_names(), help=" ")        
        parser.add_argument("--T", type=float, default=0.99, help="The biggest sample time index")
        parser.add_argument("--mlp_hidden_size", type=int, default=128, help="Number of hidden units in the MLP.")
        parser.add_argument("--transition_rate", type=float, default=1.75, help="")
        parser.add_argument("--mlp_lr", type=float, default=1e-5, help="")
        parser.add_argument("--probability_flow", action='store_true', default=False, help="probability flow")
        parser.add_argument("--corrector", type=str, choices=CorrectorRegistry.get_all_names(), default="none", help="Corrector class for the PC sampler.")
        parser.add_argument("--snr", type=float, default=0.4, help="SNR value for (annealed) Langevin dynmaics")
        parser.add_argument("--t_eps_val", type=float, default=6e-2, help="The minimum time (1e-3 by default)")
        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=3e-2, mlp_hidden_size=128, mlp_lr=1e-4,
        num_eval_files=20, loss_type='mse', data_module_cls=None, time_emb=True, seed=None,
        probability_flow=False, corrector="aldv", snr=0.5, t_eps_val=4e-2,
        p_name='reverse_diffusion', T=1.0, transition_rate=1.7, 
         **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        backbone_params = {k: v for k, v in kwargs.items() 
                        if k in inspect.signature(dnn_cls.__init__).parameters}     # 过滤出backbone需要的参数
        self.dnn = dnn_cls(**backbone_params)

        # Initialize hyperparams
        self.time_emb = time_emb
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.dnn.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.T = T
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
        self.p_name = p_name
        self.mlp_lr = mlp_lr
        self.t_eps_val = t_eps_val

        # Initialize Sampling Configurations
        self.probability_flow = probability_flow
        self.corrector = corrector
        self.snr = snr

        # Initialize SDE
        if sde == 'ic_vpsde' or sde == 'vise_vp':
            sde = 'vpvid'         # TODO: 之前训练的ckpt使用的sde名字为ic_vpsde和vise_vp，这里是为了兼容之前的代码
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)

        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        self.cal_pesq = pesq

        if(seed is not None):
            # seed = torch.randint(1, 1000000, (1,)).item()
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            pl.seed_everything(seed)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.dnn.parameters(), lr=self.lr)
        return optimizer
    
    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.dnn.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.dnn.parameters())        # store current params in EMA
                self.ema.copy_to(self.dnn.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.dnn.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, err):
        # 如果损失类型是 'mse'（均方误差），计算每个误差的平方
        if self.loss_type == 'mse':
            losses = torch.square(err.abs())
        # 如果损失类型是 'mae'（平均绝对误差），计算每个误差的绝对值
        elif self.loss_type == 'mae':
            losses = err.abs()
        
        # 取自 reduce_op 函数：对通道和位置求和，再对批次维度求均值
        # 这主要是为了得到绝对的损失值，对梯度无影响
        reduce_func = lambda x : torch.sum(x.reshape(x.shape[0], -1), dim=-1)
        loss = torch.mean(0.5*reduce_func(losses))
        return loss


    def _step(self, batch, batch_idx):
        # 关键代码
        x, y = batch

        # 生成的 t 是一个在区间 [self.t_eps, self.sde.T) 内均匀分布的随机变量
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        t, st, ut = self.sde._forward(t, z, x, y)

        model_output = self(st, t, y)    # 此处自动调用forward方法

        # velocity损失
        err = model_output - ut
        loss = self._loss(err)     # 根据具体使用MSE还是MAE损失，计算最终整个批次的损失
        return loss


    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        # print('train_loss:', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # if isinstance(self.sde, T_VPSDE_RLOSS):
        #     loss, r_loss = self._tvpsde_step(batch, batch_idx)
        #     self.log('valid_r_loss', r_loss, on_step=False, on_epoch=True, sync_dist=True)
        # else:
        #     loss = self._step(batch, batch_idx)
        # self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        # if self.global_rank == 0:
        #    torch.save(self.dnn.module.state_dict(), 'cond_on_yt_reverse.ckpt')
            
        # Evaluate speech enhancement performance
        loss = 0
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files, self.p_name,
                                                 probability_flow=self.probability_flow, corrector=self.corrector,
                                                 snr=self.snr)
            self.log('pesq', pesq, on_step=False, on_epoch=True, sync_dist=True)
            self.log('si_sdr', si_sdr, on_step=False, on_epoch=True, sync_dist=True)
            self.log('estoi', estoi, on_step=False, on_epoch=True, sync_dist=True)
    
        torch.cuda.empty_cache()

        return loss

    def forward(self, x, t, y, *args):
        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y], dim=1)
        
        # the minus is most likely unimportant here - taken from Song's repo
        model_output = self.dnn(dnn_input, t)
        return model_output

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, t_eps=None, T=None, resolution=None, minibatch=None, 
                       probability_flow=None, snr=None, **kwargs):
        N = self.sde.N if N is None else N
        T = self.sde.t if T is None else T
        snr = self.snr if snr is None else snr
        eps = self.t_eps_val if t_eps is None else t_eps
        probability_flow = self.probability_flow if probability_flow is None else probability_flow
        sde = self.sde.copy()
        sde.N = N
        sde.t = T
        sde.resolution = None if resolution is None else resolution

        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y, time_emb=self.time_emb, eps=eps,
                                            probability_flow=probability_flow, snr=snr, **kwargs)

        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini, time_emb=self.time_emb, eps=eps,
                                                      probability_flow=probability_flow, snr=snr, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns
            return batched_sampling_fn

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def enhance(self, y, sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=30, corrector_steps=1, snr=0.5, timeit=False,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        sr=16000
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        if sampler_type == "pc":
            sampler = self.get_pc_sampler(predictor, corrector, Y.cuda(), N=N, 
                corrector_steps=corrector_steps, snr=snr, intermediate=False,
                **kwargs)
        elif sampler_type == "ode":
            sampler = self.get_ode_sampler(Y.cuda(), N=N, **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))
        sample, nfe = sampler()
        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/sr)
            return x_hat, nfe, rtf
        else:
            return x_hat
