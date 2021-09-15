import torch
from models.closedform.utils import load_generator as load_cf_generator
from models.closedform.utils import load_deformator as load_cf_deformator
from models.latentdiscovery.utils import load_generator as load_ld_generator
from models.latentdiscovery.utils import load_deformator as load_ld_deformator
from models.epsilon_predictor import ResNetEpsPredictor


def get_model(config):
    if config.initialisation == 'closed_form':
        generator = load_cf_generator(config)
        deformator = load_cf_deformator(config)

    elif config.initialisation == 'latent_discovery':
        generator = load_ld_generator(config)
        deformator = load_ld_deformator(config)

    deformator_opt = torch.optim.Adam(deformator.parameters(), lr=config.deformator_lr)
    eps_predictor = ResNetEpsPredictor(num_dirs=config.num_directions).cuda()
    eps_predictor_opt = torch.optim.Adam(eps_predictor.parameters(), lr=config.eps_predictor_lr)

    return generator, deformator, deformator_opt, eps_predictor, eps_predictor_opt
