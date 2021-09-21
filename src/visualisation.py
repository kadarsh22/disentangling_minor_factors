import os
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import wandb
import torch.nn.functional as F

class Visualiser(object):
    def __init__(self, config):
        self.config = config
        self.experiment_name = wandb.run.name



    def generate_plot_save_results(self, results, plot_type):
        file_location = os.path.dirname(os.getcwd()) + f'/results/{self.experiment_name}' + '/visualisations/plots/'
        if not os.path.exists(file_location):
            os.makedirs(file_location)
        plt.figure()
        for name, values in results.items():
            x_axis = list(range(len(values)))
            plt.plot(x_axis, values, label=name)
        plt.legend(loc="upper right")
        path = file_location + str(plot_type) + '.jpeg'
        plt.savefig(path)

    def visualise_directions(self, generator, deformator, iteration):
        directions = deformator.ortho_mat
        directions_artifact = wandb.Artifact("test_samples_" + str(wandb.run.id), type="predictions")
        directions_table = wandb.Table(columns=['images', 'direction_idx'])
        for dir_idx in range(self.config.batch_size):
            direction = directions[dir_idx: dir_idx + 1]
            images_shifted = generator(direction)
            images_shifted = (images_shifted + 1) / 2
            image = F.avg_pool2d(images_shifted, 4, 4)
            image = torch.randn((3,128,128))
            directions_table.add_data(wandb.Image(image), str(dir_idx))
        directions_artifact.add(directions_table, "predictions")
        wandb.run.log_artifact(directions_artifact)
        print("-------------------------logged_once------------------------------------------")


    def visualise_latent_traversal(self, initial_rep, decoder, epoch_num):
        interval_start = self.config['interval_start']
        interval = (2 * (interval_start)) / 10
        interpolation = torch.arange(-1 * interval_start, interval_start + interval, interval)
        rep_org = initial_rep
        file_location = os.path.dirname(
            os.getcwd()) + f'/results/{self.experiment_name}' + '/visualisations/latent_traversal/'
        if not os.path.exists(file_location):
            os.makedirs(file_location)
        path = file_location + str(epoch_num) + '.jpeg'
        samples = []
        z_ = torch.rand((1, self.config['noise_dim'])).cuda()
        for j in range(self.config['latent_dim']):
            temp = initial_rep.data[:, j].clone()
            for k in interpolation:
                rep_org.data[:, j] = k
                if self.config['model_arch'] == 'gan':
                    final_rep = torch.cat((z_, rep_org), dim=1)
                    sample = decoder(final_rep)  # TODO need not be sigmoid
                else:
                    sample = torch.sigmoid(decoder(rep_org))
                sample = sample.view(-1, 64, 64)
                samples.append(sample)
            rep_org.data[:, j] = temp
        grid_img = torchvision.utils.make_grid(samples, nrow=10, padding=10, pad_value=1)
        grid = grid_img.permute(1, 2, 0).type(torch.FloatTensor)
        plt.imsave(path, grid.data.numpy())
