import torch
from models.closedform.utils import load_generator as load_cf_generator
from models.closedform.utils import load_deformator as load_cf_deformator
from models.latentdiscovery.utils import load_generator as load_ld_generator
from models.latentdiscovery.utils import load_deformator as load_ld_deformator


def get_model(opt):
    if opt.algo.ours.initialisation == 'closed_form':
        generator = load_cf_generator(opt)
        deformator = load_cf_deformator(opt)

    elif opt.algo.ours.initialisation == 'latent_discovery':
        generator = load_ld_generator(opt)
        deformator = load_ld_deformator(opt)

    deformator_opt = torch.optim.Adam(deformator.parameters(), lr=opt.algo.ours.deformator_lr)

    return generator, deformator, deformator_opt