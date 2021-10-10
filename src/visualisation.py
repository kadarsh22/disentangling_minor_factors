import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch.nn.functional as F


class Visualiser(object):
    def __init__(self, config):
        self.config = config
        self.experiment_name = wandb.run.name

    def visualise_directions(self, generator, deformator, iteration):
        directions = deformator.ortho_mat
        directions_artifact = wandb.Artifact(str(wandb.run.name) + str('_directions'), type="Directions")
        directions_table = wandb.Table(columns=['images', 'direction_idx'])
        for dir_idx in range(self.config.batch_size):
            direction = directions[dir_idx: dir_idx + 1]
            images_shifted = generator.generator(generator.generator.gen.style(direction.view(-1, self.config.latent_dim)))
            images_shifted = (images_shifted + 1) / 2
            image = F.avg_pool2d(images_shifted, 4, 4)
            directions_table.add_data(wandb.Image(image), str(dir_idx))
        directions_artifact.add(directions_table, "directions")
        wandb.run.log_artifact(directions_artifact, aliases=[str(iteration)])

    def postprocess_images(self, images):
        """Post-processes images from `torch.Tensor` to `numpy.ndarray`."""
        images = images.detach().cpu().numpy()
        images = (images + 1) * 255 / 2
        images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
        images = images.transpose(0, 2, 3, 1)
        return images

    def generate_latent_traversal(self, generator, deformator, iteration ,seed):
        min_index = 0
        directions = deformator.ortho_mat
        temp_path = os.path.join(self.config.result_path, 'temp')
        os.makedirs(temp_path, exist_ok=True)
        z = generator.sample_zs(self.config.batch_size, seed)
        latent_traversal_artifact = wandb.Artifact(str(wandb.run.name) + '_latent_traversals', type="Latent Traversals")
        lt_table = wandb.Table(columns=['image_grid', 'direction_idx'])

        for dir_idx in range(self.config.eval_directions):
            shifted_w = []
            for idx, z_ in enumerate(z):
                for i, shift in enumerate(
                        np.linspace(-self.config.shifts_r, self.config.shifts_r, self.config.shifts_count)):
                    w = generator.generator.gen.style(z_)
                    shifted_w.append(w + directions[dir_idx: dir_idx + 1] * shift)
            shifted_w = torch.stack(shifted_w).squeeze(dim=1)
            with torch.no_grad():
                cf_images = torch.stack([F.avg_pool2d(generator.generator(shifted_w[idx].view(-1, 512)), 16, 16) for idx in
                                         range(shifted_w.shape[0])]).view(-1, 3, 64, 64)
            grid = torchvision.utils.make_grid(cf_images.clamp(min=-1, max=1), nrow=3, scale_each=True, normalize=True)
            lt_table.add_data(wandb.Image(grid), str(dir_idx))
            plt.imsave(os.path.join(temp_path, str(min_index) + '.png'), grid.permute(1, 2, 0).cpu().numpy())
            min_index = min_index + 1
        latent_traversal_artifact.add(lt_table, "lt")
        wandb.run.log_artifact(latent_traversal_artifact, aliases=str(iteration))
