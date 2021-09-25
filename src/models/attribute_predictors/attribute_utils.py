import torch
import os
from . import attribute_predictor_gan_ensemble

softmax = torch.nn.Softmax(dim=1)


def downsample(images, size=256):
    # Downsample to 256x256. The attribute classifiers were built for 256x256.
    # follows https://github.com/NVlabs/stylegan/blob/master/metrics/linear_separability.py#L127
    if images.shape[2] > size:
        factor = images.shape[2] // size
        assert (factor * size == images.shape[2])
        images = images.view(
            [-1, images.shape[1], images.shape[2] // factor, factor, images.shape[3] // factor, factor])
        images = images.mean(dim=[3, 5])
        return images
    else:
        assert (images.shape[-1] == 256)
        return images


def get_logit(net, im):
    im_256 = downsample(im)
    logit = net(im_256)
    return logit


def get_linear_out(net, im):
    logit = get_logit(net, im)
    linear_out = torch.cat([logit, -logit], dim=1)
    # logit is (N,) softmaxed is (N,)
    return linear_out


def load_attribute_classifier(attribute, ckpt_path=None, device='cuda'):
    # if ckpt_path is None:
    #     base_path = '/home/adarsh/PycharmProjects/disentagled_latent_dirs/pretrained_models/classifiers/celebahq'
    #     attribute_pkl = os.path.join(base_path, attribute, 'net_best.pth')
    #     ckpt = torch.load(attribute_pkl)
    # else:
    #     ckpt = torch.load(ckpt_path)
    attribute_pkl = os.path.join(ckpt_path, attribute, 'net_best.pth')
    ckpt = torch.load(attribute_pkl, map_location=device)
    # print("Using classifier at epoch: %d" % ckpt['epoch'])
    # if 'valacc' in ckpt.keys():
    #     print("Validation acc on raw images: %0.5f" % ckpt['valacc'])
    detector = attribute_predictor_gan_ensemble.from_state_dict(
        ckpt['state_dict'], fixed_size=True, use_mbstd=False).to(device).eval()
    return detector


class ClassifierWrapper(torch.nn.Module):
    def __init__(self, classifier_name, ckpt_path=None, device='cuda'):
        super(ClassifierWrapper, self).__init__()
        self.net = load_attribute_classifier(classifier_name, ckpt_path, device).eval().to(device)

    @torch.no_grad()
    #     @torch.cuda.amp.autocast()
    def forward(self, ims):
        # returns (N,) softmax values for binary classification
        return get_linear_out(self.net, ims)
