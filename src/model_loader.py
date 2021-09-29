import torch
from models.closedform.utils import load_generator as load_cf_generator
from models.closedform.utils import load_deformator as load_cf_deformator
from models.latentdiscovery.utils import load_generator as load_ld_generator
from models.latentdiscovery.utils import load_deformator as load_ld_deformator
from models.epsilon_predictor import ResNetEpsPredictor
from models import domain_generator
from models import latent_regressor

BB_KWARGS = {
    "shapes3d": {"in_channel": 3, "size": 64},
    "mpi3d": {"in_channel": 3, "size": 64},
    # grayscale -> rgb
    "dsprites": {"in_channel": 1, "size": 64},
    "cars3d": {"in_channel": 3, "size": 64, "f_size": 512},
    "isaac": {"in_channel": 3, "size": 128, "f_size": 512},
}


def get_model(config):
    if config.initialisation == 'closed_form':
        generator = load_cf_generator(config)
        deformator = load_cf_deformator(config)

    elif config.initialisation == 'latent_discovery':
        generator = load_ld_generator(config)
        deformator = load_ld_deformator(config)

    generator = domain_generator.define_generator('stylegan2', 'celebahq')

    inversion_network = latent_regressor.Encoder(latent_dimension=config.latent_dim,
                                                 backbone="cnn_encoder", **BB_KWARGS['isaac'])
    deformator_opt = torch.optim.Adam(deformator.parameters(), lr=config.deformator_lr)
    eps_predictor = ResNetEpsPredictor(num_dirs=config.num_directions).to(config.device)
    eps_predictor_opt = torch.optim.Adam(eps_predictor.parameters(), lr=config.eps_predictor_lr)

    return generator, deformator, deformator_opt, eps_predictor, eps_predictor_opt, inversion_network
