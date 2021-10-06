import torch
from models.closedform.utils import load_generator as load_cf_generator
from models.closedform.utils import load_deformator as load_cf_deformator
from models.latentdiscovery.utils import load_generator as load_ld_generator
from models.latentdiscovery.utils import load_deformator as load_ld_deformator
from models.latent_shift_predictor import LatentShiftPredictor
from models.closedform.closedform_directions import CfOrtho


def get_model(config):
    if config.initialisation == 'closed_form':
        source_generator = load_cf_generator(config.source_model_name, config.device)
        source_deformator = load_cf_deformator(config, config.source_model_name)
        target_generator = load_cf_generator(config.target_model_name, config.device)
        target_deformator = CfOrtho(config.latent_dim, config.num_directions)

    elif config.initialisation == 'latent_discovery':
        generator = load_ld_generator(config)
        deformator = load_ld_deformator(config)

    transformation_learning_net = LatentShiftPredictor(config.num_output_units)
    transformation_learning_net_opt = torch.optim.Adam(
        transformation_learning_net.parameters(), lr=config.transformation_learning_net_lr)
    target_deformator_opt = torch.optim.Adam(target_deformator.parameters(), lr=config.target_deformator_lr)
    return source_generator, source_deformator, target_generator, target_deformator, target_deformator_opt, \
           transformation_learning_net, transformation_learning_net_opt
