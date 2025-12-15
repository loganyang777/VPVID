import wandb, os

import argparse
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from os.path import join

# Set CUDA architecture list
from sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.model import ScoreModel


def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups


if __name__ == '__main__':
     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
          parser_.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve")
          parser_.add_argument("--nolog", action='store_true', help="Turn off logging.")
          parser_.add_argument("--no_wandb", action='store_true', help="Turn off logging to W&B, using local default logger instead")
          parser_.add_argument("--wandb_name", type=str, default=None, help="Name for wandb logger. If not set, a random name is generated.")
          parser_.add_argument("--wandb_project", type=str, default="sgmse", help="Project name for wandb logger.")
          parser_.add_argument("--ckpt", type=str, default=None, help="Resume training from checkpoint.")
          parser_.add_argument("--logdir", type=str, default=None, help="Directory to save logs.")
          # parser_.add_argument("--resume_from_checkpoint", type=str, default='none', help="Turn off logging to W&B, using local default logger instead")

     temp_args, _ = base_parser.parse_known_args()

     # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
     backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
     sde_class = SDERegistry.get_by_name(temp_args.sde)
     trainer_parser = parser.add_argument_group("Trainer", description="Lightning Trainer")
     trainer_parser.add_argument("--accelerator", type=str, default="gpu", help="Supports passing different accelerator types.")
     trainer_parser.add_argument("--devices", default="auto", help="How many gpus to use.")
     # trainer_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients.")
     trainer_parser.add_argument("--max_epochs", type=int, default=200, help="Maximum number of epochs to train.")
     
     ScoreModel.add_argparse_args(
          parser.add_argument_group("ScoreModel", description=ScoreModel.__name__))
     sde_class.add_argparse_args(
          parser.add_argument_group("SDE", description=sde_class.__name__))
     backbone_cls.add_argparse_args(
          parser.add_argument_group("Backbone", description=backbone_cls.__name__))
     # Add data module args
     data_module_cls = SpecsDataModule
     data_module_cls.add_argparse_args(
          parser.add_argument_group("DataModule", description=data_module_cls.__name__))
     # Parse args and separate into groups
     args = parser.parse_args() 
     arg_groups = get_argparse_groups(parser)
 
     logdir = args.logdir
     logdir = 'logs' if logdir == None else os.path.join(logdir, 'logs')

     # Initialize logger, trainer, model, datamodule
     model = ScoreModel(
          backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
          **{
               **vars(arg_groups['ScoreModel']),
               **vars(arg_groups['SDE']),
               **vars(arg_groups['Backbone']),
               **vars(arg_groups['DataModule'])
          }
     )

     # Set up logger configuration
     if args.nolog:
          logger = None
     else:
          if args.no_wandb:
               logger = TensorBoardLogger(save_dir=logdir, name="tensorboard")
          else:
               logger = WandbLogger(project=args.wandb_project, log_model=True, save_dir="logs", name=args.wandb_name)
               logger.experiment.log_code(".")

     # Set up callbacks for logger
     callbacks = [ModelCheckpoint(dirpath=f"{logdir}/{logger.version}", save_last=True, filename='{epoch}-last')]
     if args.num_eval_files:
          checkpoint_callback_pesq = ModelCheckpoint(dirpath=f"{logdir}/{logger.version}", 
               save_top_k=2, monitor="pesq", mode="max", filename='{epoch}-{pesq:.2f}')
          checkpoint_callback_si_sdr = ModelCheckpoint(dirpath=f"{logdir}/{logger.version}", 
               save_top_k=2, monitor="si_sdr", mode="max", filename='{epoch}-{si_sdr:.2f}')
          callbacks += [checkpoint_callback_pesq, checkpoint_callback_si_sdr]
     # resume_from_checkpoint = None if args.resume_from_checkpoint == 'none' else args.resume_from_checkpoint
     # Initialize the Trainer and the DataModule
     trainer = pl.Trainer(
          **vars(arg_groups['Trainer']),
          strategy=DDPStrategy(find_unused_parameters=False), logger=logger,
          log_every_n_steps=20, num_sanity_val_steps=0, check_val_every_n_epoch=1,
          callbacks=callbacks,
          # resume_from_checkpoint=resume_from_checkpoint,
     )

     # Train model
     trainer.fit(model, ckpt_path=args.ckpt)