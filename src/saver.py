import os
import torch
import numpy as np
import random
import wandb


class Saver(object):
    def __init__(self, config):
        self.config = config

    def save_model(self, params, step):
        cwd = os.path.dirname('results/'+ str(wandb.run.name)+'/')  # project root
        models_dir = cwd + '/models/'

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        deformator, deformator_opt, eps_predictor, eps_predictor_opt = params

        torch.save({
            'step': step,
            'deformator': deformator.state_dict(),
            'deformator_opt': deformator_opt.state_dict(),
            'eps_predictor': eps_predictor.state_dict(),
            'eps_opt': eps_predictor_opt.state_dict(),
            'torch_rng_state': torch.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'random_state': random.getstate()

        }, os.path.join(models_dir, str(step) + '_model.pkl'))
        deformator.ortho_mat.data[0][0] = deformator.ortho_mat.data[0][0] / 2
        artifact = wandb.Artifact(wandb.run.name, type='model')
        artifact.add_file(os.path.join(models_dir, str(step) + '_model.pkl'))
        wandb.run.log_artifact(artifact)
        return True

    def load_model(self, params):
        models_dir = os.path.dirname(os.getcwd()) + f'/results/{wandb.run.name}' + '/models/' + self.config[
            'file_name']  # project root
        checkpoint = torch.load(models_dir)
        run = wandb.init()
        artifact = run.use_artifact('entity/your-project-name/model:v0', type='model')
        artifact_dir = artifact.download()
        run.join()

        deformator, deformator_opt = params
        deformator.load_state_dict(checkpoint['deformator'])
        deformator_opt.load_state_dict(checkpoint['deformator_opt'])
        torch.set_rng_state(checkpoint['torch_rng_state'])
        np.random.set_state(checkpoint['np_rng_state'])
        random.setstate(checkpoint['random_state'])
        return deformator, deformator_opt
