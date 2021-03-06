# -*- coding: utf-8 -*-
"""
Training MulMON on a multiple GPU devices.
@author: Nanbo Li
"""
import sys
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.distributed as distributed
import torch.multiprocessing as mp

# set project search path
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)

from config import CONFIG
from scheduler import AnnealingStepLR
from trainer.model_trainer_parallel import ModelTrainer
from utils import set_random_seed, load_trained_mp, ensure_dir


# ------------------------- respecify important flags ------------------------
def running_cfg(cfg):
    ###########################################
    # Config i/o path
    ###########################################
    if cfg.DATA_TYPE == 'gqn_jaco':
        image_size = [64, 64]
        CLASSES = ['_background_', 'jaco', 'generic']
        cfg.v_in_dim = 7
        cfg.max_sample_views = 6
        data_dir = cfg.DATA_ROOT
        assert os.path.exists(data_dir)
        train_data_filename = os.path.join(data_dir, 'gqn_jaco', 'gqn_jaco_train.h5')
        test_data_filename = os.path.join(data_dir, 'gqn_jaco', 'gqn_jaco_test.h5')
        assert os.path.isfile(train_data_filename)
        assert os.path.isfile(test_data_filename)
    elif cfg.DATA_TYPE == 'clevr_mv':
        image_size = [64, 64]
        CLASSES = ['_background_', 'cube', 'sphere', 'cylinder']
        cfg.v_in_dim = 3
        cfg.max_sample_views = 6
        data_dir = cfg.DATA_ROOT
        assert os.path.exists(data_dir)
        train_data_filename = os.path.join(data_dir, 'clevr_mv', 'clevr_mv_train.json')
        test_data_filename = os.path.join(data_dir, 'clevr_mv', 'clevr_mv_test.json')
        assert os.path.isfile(train_data_filename)
        assert os.path.isfile(test_data_filename)
    elif cfg.DATA_TYPE == 'clevr_aug':
        image_size = [64, 64]
        CLASSES = ['_background_', 'diamond', 'duck', 'mug', 'horse', 'dolphin']
        cfg.v_in_dim = 3
        cfg.max_sample_views = 6
        data_dir = cfg.DATA_ROOT
        assert os.path.exists(data_dir)
        train_data_filename = os.path.join(data_dir, 'clevr_aug', 'clevr_aug_train.json')
        test_data_filename = os.path.join(data_dir, 'clevr_aug', 'clevr_aug_test.json')
        assert os.path.isfile(train_data_filename)
        assert os.path.isfile(test_data_filename)
    # ------------------- For your customised CLEVR -----------------------
    elif cfg.DATA_TYPE == 'your-clevr':
        image_size = [64, 64]
        CLASSES = ['_background_', 'xxx']
        cfg.v_in_dim = 3
        cfg.max_sample_views = 6
        data_dir = cfg.DATA_ROOT
        assert os.path.exists(data_dir)
        train_data_filename = os.path.join(data_dir, 'your-clevr', 'your-clevr_train.json')
        test_data_filename = os.path.join(data_dir, 'your-clevr', 'your-clevr_test.json')
        assert os.path.isfile(train_data_filename)
        assert os.path.isfile(test_data_filename)
    # ------------------- For your customised CLEVR -----------------------
    else:
        raise NotImplementedError

    cfg.view_dim = cfg.v_in_dim

    # log directory
    ckpt_base = cfg.ckpt_base
    ensure_dir(ckpt_base)

    # model savedir
    check_dir = os.path.join(ckpt_base, '{}_log/'.format(cfg.arch))
    ensure_dir(check_dir)

    # generated sample dir
    save_dir = os.path.join(check_dir, 'saved_models/')
    ensure_dir(save_dir)

    # visualise training epochs
    vis_train_dir = os.path.join(check_dir, 'vis_training/')
    ensure_dir(vis_train_dir)

    # generated sample dir  (for testing generation)
    generated_dir = os.path.join(check_dir, 'generated/')
    ensure_dir(generated_dir)

    if cfg.resume_path is not None:
        assert os.path.isfile(cfg.resume_path)
    elif cfg.resume_epoch is not None:
        resume_path = os.path.join(save_dir,
                                   'checkpoint-epoch{}.pth'.format(cfg.resume_epoch))
        assert os.path.isfile(resume_path)
        cfg.resume_path = resume_path

    cfg.DATA_DIR = data_dir
    cfg.train_data_filename = train_data_filename
    cfg.test_data_filename = test_data_filename
    cfg.check_dir = check_dir
    cfg.save_dir = save_dir
    cfg.vis_train_dir = vis_train_dir
    cfg.generated_dir = generated_dir

    cfg.image_size = image_size
    cfg.CLASSES = CLASSES
    cfg.num_classes = len(CLASSES)

    return cfg


# ---------------------------- main function -----------------------------
def get_trainable_params(model):
    params_to_update = []
    print('trainable parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("\t", name)
            params_to_update.append(param)
    return params_to_update


def train(process_id, CFG):
    if 'GQN' in CFG.arch:
        from models.baseline_gqn import GQN as ScnModel
        print(" --- Arch: GQN ---")
    elif 'IODINE' in CFG.arch:
        from models.baseline_iodine import IODINE as ScnModel
        print(" --- Arch: IODINE ---")
    elif 'MulMON' == CFG.arch:
        from models.mulmon import MulMON as ScnModel
        print(" --- Arch: MulMON ---")
    elif 'FastMulMON' == CFG.arch:
        from models.fast_mulmon import FastMulMON as ScnModel
        print(" --- Arch: FastMulMON ---")
    else:
        raise NotImplementedError

    rank = CFG.nrank * CFG.gpus + process_id
    gpu = process_id + CFG.gpu_start   # e.g. gpus=2, gpu_start=1 means using gpu [0+1, 1+1] = [1, 2].

    distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=CFG.world_size,
            rank=rank
    )

    if CFG.seed is None:
        CFG.seed = random.randint(0, 1000000)
    set_random_seed(CFG.seed)

    # Create the model
    scn_model = ScnModel(CFG)
    torch.cuda.set_device(gpu)

    if CFG.resume_epoch is not None:
        state_dict = load_trained_mp(CFG.resume_path)
        scn_model.load_state_dict(state_dict, strict=True)

    scn_model.cuda(gpu)
    params_to_update = get_trainable_params(scn_model)

    if CFG.optimiser == 'RMSprop':
        optimiser = torch.optim.RMSprop(params_to_update,
                                        lr=CFG.lr_rate,
                                        weight_decay=CFG.weight_decay)
        lr_scheduler = None
    else:
        optimiser = torch.optim.Adam(params_to_update,
                                     lr=CFG.lr_rate,
                                     weight_decay=CFG.weight_decay)
        lr_scheduler = AnnealingStepLR(optimiser, mu_i=CFG.lr_rate, mu_f=0.1*CFG.lr_rate, n=1.0e6)

    scn_model = nn.parallel.DistributedDataParallel(scn_model,
                                                    device_ids=[gpu])

    if 'gqn' in CFG.DATA_TYPE:
        from data_loader.getGqnH5 import distributed_loader
    elif 'clevr' in CFG.DATA_TYPE:
        from data_loader.getClevrMV import distributed_loader
    else:
        raise NotImplementedError

    train_dataset = distributed_loader(CFG.DATA_ROOT,
                                       CFG.train_data_filename,
                                       num_slots=CFG.num_slots,
                                       use_bg=CFG.use_bg)
    val_dataset = distributed_loader(CFG.DATA_ROOT,
                                     CFG.test_data_filename,
                                     num_slots=CFG.num_slots,
                                     use_bg=CFG.use_bg)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=CFG.world_size,
        rank=rank
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=CFG.world_size,
        rank=rank
    )

    # get data Loader
    train_dl = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=CFG.batch_size,
                                           shuffle=False,
                                           num_workers=0,
                                           pin_memory=True,
                                           collate_fn=lambda x: tuple(zip(*x)),
                                           sampler=train_sampler)
    val_dl = torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=CFG.batch_size,
                                         shuffle=False,
                                         num_workers=0,
                                         pin_memory=True,
                                         collate_fn=lambda x: tuple(zip(*x)),
                                         sampler=val_sampler)
    trainer = ModelTrainer(
        model=scn_model,
        loss=None,
        metrics=None,
        optimizer=optimiser,
        step_per_epoch=CFG.step_per_epoch,
        config=CFG,
        train_data_loader=train_dl,
        valid_data_loader=val_dl,
        device=gpu,
        lr_scheduler=lr_scheduler
    )
    # Start training session
    trainer.train()


def main(cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='ScnModel',
                        help="model name")
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--datatype', type=str, default='clevr',
                        help="one of [gqn_jaco, clevr_mv, clevr_aug]")
    parser.add_argument('--epochs', default=1000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--step_per_epoch', default=0, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=4, type=int, metavar='N', help='number of data samples of a minibatch')
    parser.add_argument('--work_mode', type=str, default='training', help="model's working mode")
    parser.add_argument('--optimiser', type=str, default='Adam', help="help= one of [Adam, RMSprop]")
    parser.add_argument('--resume_epoch', default=None, type=int, metavar='N',
                        help='resume weights from [N]th epochs')

    parser.add_argument('--nodes', default=1, type=int, metavar='N')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('--nrank', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--gpu_start', default=0, type=int, help='first gpu indicator, default using 0 as the start')
    parser.add_argument('--master_port', default='8888', type=str, help='used for rank0 communication with others')

    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--lr_rate', default=1e-4, type=float, help='learning rate')
    # Patrick
    parser.add_argument('--refinement_iters', default=3, type=int, help='# iters of inner inference loop')
    parser.add_argument('--stochastic_layers', default=3, type=int, help='# of stoch layers in HVAE')
    parser.add_argument('--num_slots', default=7, type=int, help='(maximum) number of component slots')
    parser.add_argument('--temperature', default=0.0, type=float,
                        help='spatial scheduler increase rate, the hotter the faster coeff grows')
    parser.add_argument('--latent_dim', default=16, type=int, help='size of the latent dimensions')
    parser.add_argument('--view_dim', default=5, type=int, help='size of the viewpoint latent dimensions')
    parser.add_argument('--min_sample_views', default=1, type=int, help='mininum allowed #views for scene learning')
    parser.add_argument('--max_sample_views', default=5, type=int, help='maximum allowed #views for scene learning')
    parser.add_argument('--num_vq_show', default=5, type=int, help='#views selected for visualisation')
    parser.add_argument('--pixel_sigma', default=0.1, type=float, help='loss strength item')
    parser.add_argument('--kl_latent', default=1.0, type=float, help='loss strength item')
    parser.add_argument('--kl_spatial', default=1.0, type=float, help='loss strength item')
    parser.add_argument('--exp_attention', default=1.0, type=float, help='loss strength item')
    parser.add_argument('--query_nll', default=1.0, type=float, help='loss strength item')
    parser.add_argument('--exp_nll', default=1.0, type=float, help='loss strength item')

    parser.add_argument("--use_mask", help="use gt mask to by pass the segmentation phase",
                        action="store_true", default=False)
    parser.add_argument("--use_bg", help="treat background as an object",
                        action="store_true", default=False)

    parser.add_argument("-i", '--input_dir', required=True,  help="path to the input data for the model to read")
    parser.add_argument("-o", '--output_dir', required=True,  help="destination dir for the model to write out results")
    args = parser.parse_args()

    ###########################################
    # General training reconfig
    ###########################################
    cfg.arch = args.arch
    # Patrick
    cfg.run_name = args.run_name
    if cfg.run_name == '':
        cfg.run_name = cfg.arch
    cfg.DATA_TYPE = args.datatype
    cfg.num_epochs = args.epochs
    cfg.step_per_epoch = args.step_per_epoch if args.step_per_epoch > 0 else None
    cfg.batch_size = args.batch_size
    cfg.WORK_MODE = args.work_mode
    cfg.optimiser = args.optimiser
    cfg.resume_epoch = args.resume_epoch
    cfg.seed = args.seed
    cfg.lr_rate = args.lr_rate
    cfg.num_slots = args.num_slots
    cfg.temperature = args.temperature
    cfg.latent_dim = args.latent_dim
    cfg.view_dim = args.view_dim
    cfg.min_sample_views = args.min_sample_views
    cfg.max_sample_views = args.max_sample_views
    cfg.num_vq_show = args.num_vq_show
    cfg.pixel_sigma = args.pixel_sigma
    cfg.use_mask = args.use_mask
    cfg.use_bg = args.use_bg
    # Patrick
    cfg.stochastic_layers = args.stochastic_layers
    cfg.refinement_iters = args.refinement_iters
    cfg.elbo_weights = {
        'kl_latent': args.kl_latent,
        'kl_spatial': args.kl_spatial,
        'exp_attention': args.exp_attention,
        'exp_nll': args.exp_nll,
        'query_nll': args.query_nll
    }
    # I/O path configurations
    cfg.DATA_ROOT = args.input_dir
    cfg.ckpt_base = args.output_dir

    ###########################################
    # Config gpu usage
    ###########################################
    cfg.nodes = args.nodes
    cfg.gpus = args.gpus
    cfg.nrank = args.nrank
    cfg.gpu_start = args.gpu_start
    cfg.world_size = args.gpus * args.nodes  #

    running_cfg(cfg)
    os.environ['MASTER_ADDR'] = 'localhost'  #
    os.environ['MASTER_PORT'] = args.master_port  #
    mp.spawn(train, nprocs=cfg.gpus, args=(cfg,))


##############################################################################
if __name__ == "__main__":
    cfg = CONFIG()
    main(cfg)
