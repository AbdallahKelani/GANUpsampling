import torch
import pickle
import os
import sys
import random
import numpy as np
from PIL import Image
from torch.optim import Optimizer
import gradio as gr

from ranger import Ranger21

os.sys.path.append(os.getcwd() +'/stylegan3')

optimizers = {"Adam": torch.optim.Adam,
              "SGD": torch.optim.SGD,
              "Ranger21": Ranger21}


# Using the slightly modified loss and optimizer implementations from
# https://github.com/adamian98/pulse

def loss(latent, img, ref_img, c_l2, c_ld):
    #img_32.detach()
    X = latent.view(-1, 1, 14, 512)
    Y = latent.view(-1, 14, 1, 512)
    A = ((X-Y).pow(2).sum(-1)+1e-9).sqrt()
    B = ((X+Y).pow(2).sum(-1)+1e-9).sqrt()
    D = 2*torch.atan2(A, B)
    D = ((D.pow(2)*512).mean((1, 2))/8.).sum()
    
    #l1 = 10 * ((img - ref_img).abs().mean((1, 2, 3)).clamp(min=2e-3).sum())
    l2 = ((img - ref_img).pow(2).mean((1, 2, 3)).clamp(min=2e-3).sum())
    geodesic = D
    return c_l2 * (l2) + c_ld * geodesic

class SphericalOptimizer(Optimizer):
    def __init__(self, optimizer, params, **kwargs):
        self.opt = optimizer(params, **kwargs)
        self.params = params
        with torch.no_grad():
            self.radii = {param: (param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt() for param in params}

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)
        for param in self.params:
            param.data.div_((param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt())
            param.mul_(self.radii[param])

        return loss

def downsample(img):
    img_32 = torch.nn.functional.interpolate(img, size=(32, 32), mode='bicubic')[0]
    img_32 = torch.clamp(img_32 / 255, 0, 1)
    return img_32

model_name = "stylegan2-ffhq-256x256.pkl"
with open(model_name, 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module

for param in G.parameters():
    param.requires_grad = False

gaussian_fit = torch.load("gaussian_fit_ffhq256.pt", map_location=torch.device('cuda:0'))
leaky = torch.nn.LeakyReLU(negative_slope=0.025)

def add_gaussian_noise(img, mean, stddev, mask=None):
    noise = np.random.normal(mean, stddev, img.shape).astype(np.float32)
    noise = torch.from_numpy(noise).to(img.device)
    if mask is None:
        noisy_img = torch.clamp(img + noise, 0, 1)
        return noisy_img
    
    mask = np.float32(np.all(mask[:, :, :3] == 255, axis=2)) * 255
    mask = np.expand_dims(mask, axis=0)
    mask = np.repeat(mask, 3, axis=0)
    mask = np.expand_dims(mask, axis=0)
    mask = torch.tensor(mask).cuda()
    mask = downsample(mask)

    noisy_img = torch.clamp(img + noise, 0, 1)
    final_img = noisy_img * mask + img * (1 - mask)
    
    return final_img


def generate(img, optimizer, ds_method, ds_size, lr, steps, 
             batch_size, c_l2, c_ld, noise_stddev, seed, add_noise=False):

    if seed != 0:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

    ds_size = (ds_size, ds_size)

    mask = None
    if type(img) is dict:
        mask = img['mask']
        img = img['image']
        add_noise = True

    ref = img.astype(np.float32)
    ref = np.expand_dims(ref, axis=0)
    ref = np.transpose(ref, (0, 3, 1, 2))
    ref = torch.tensor(ref).cuda()
    ref_32 = torch.nn.functional.interpolate(ref, size=ds_size, mode=ds_method)
    ref_32 = torch.clamp(ref_32 / 255, 0, 1)
    
    if add_noise:
        ref_32 = add_gaussian_noise(ref_32, 0, noise_stddev, mask)
    
    ref_32 = ref_32.detach()
    latent = torch.randn((batch_size, 14, 512), dtype=torch.float, requires_grad=True, device='cuda')
    #z = torch.from_numpy(np.random.randn(batch_size, G.z_dim)).cuda()
    #latent2 = G.mapping(z, None, truncation_psi=0.5, truncation_cutoff=8)
    #latent = G.mapping(z, None, truncation_psi=0.5, truncation_cutoff=8)
    #latent.requires_grad = True
    #latent = torch.tensor((0.9 * latent.detach() + 0.1 * latent2.detach()), dtype=torch.float, requires_grad=True, device='cuda')
    #optim = SphericalOptimizer(optimizers[optimizer], [latent], lr=lr)
    opt_settings = {"lr": lr}
    if optimizer == "Ranger21":
        opt_settings["num_epochs"] = steps
        opt_settings["num_batches_per_epoch"] = batch_size
    optim = SphericalOptimizer(optimizers[optimizer], [latent], **opt_settings)
    #optim = SphericalOptimizer(Ranger21, [latent], lr=lr, num_epochs=steps, num_batches_per_epoch=batch_size)

    # Schedulers from https://github.com/adamian98/pulse
    #lr_schedule = 'linear1cycledrop'
    #schedule_dict = {
    #            'fixed': lambda x: 1,
    #            'linear1cycle': lambda x: (9*(1-np.abs(x/steps-1/2)*2)+1)/10,
    #            'linear1cycledrop': lambda x: (9*(1-np.abs(x/(0.9*steps)-1/2)*2)+1)/10 if x < 0.9*steps else 1/10 + (x-0.9*steps)/(0.1*steps)*(1/1000-1/10),
    #        }
    #schedule_func = schedule_dict[lr_schedule]
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optim.opt, schedule_func)

    min_loss = [np.inf] * batch_size
    best = [None] * batch_size
    best_32 = [None] * batch_size
    for i in range(steps):
        optim.opt.zero_grad()
        for b in range(batch_size):
            w = leaky(latent[None, b] * gaussian_fit["std"] + gaussian_fit["mean"])
            img = (G.synthesis(w, noise_mode='random', force_fp32=True) + 1) / 2 # NCHW, float32, dynamic range [-1, +1], no truncation
            #img = (G.synthesis(w, noise_mode='none', force_fp32=True) + 1) / 2 # NCHW, float32, dynamic range [-1, +1], no truncation
            img = img.clamp(0, 1)
            img_32 = torch.nn.functional.interpolate(img, size=ds_size, mode=ds_method)
            img_32 = img_32.clamp(0, 1)
            l = loss(w, img_32, ref_32, c_l2, c_ld)
            print(l, end='\r')
            if (l < min_loss[b]):
                with torch.no_grad():
                    min_loss[b] = l.detach()
                    best[b] = img.clone().detach()
                    best_32[b] = img_32.clone().detach()
            l.backward()
        optim.step()
    #    scheduler.step()

    best_32 = [x.cpu().numpy().transpose((0, 2, 3, 1))[0] for x in best_32]
    best = [x.cpu().numpy().transpose((0, 2, 3, 1))[0] for x in best]

    return ref_32.cpu().numpy().transpose((0, 2, 3, 1))[0], \
           best_32, \
           best, \
           l.item()

css = '''
.ds_img {
    image-rendering: optimizeSpeed !important;             /*                     */
    image-rendering: -moz-crisp-edges !important;          /* Firefox             */
    image-rendering: -o-crisp-edges !important;            /* Opera               */
    image-rendering: -webkit-optimize-contrast !important; /* Chrome (and Safari) */
    image-rendering: pixelated !important;                 /* Chrome as of 2019   */
    image-rendering: optimize-contrast !important;         /* CSS3 Proposed       */
    -ms-interpolation-mode: nearest-neighbor !important;   /* IE8+                */
}

#left {
    flex-grow: 23 !important;
}

#right {
    flex-grow: 26 !important;
}

#imgs_left {
    flex-grow: 0 !important;
    min-width: 256px !important;
}
'''

with gr.Blocks(css=css) as app:
    with gr.Tab("Generate"):
        with gr.Row():
            with gr.Column(elem_id="left"):
                inp_img = gr.Image(shape=None)
                with gr.Row():
                    ds_method = gr.Dropdown(["nearest", "bilinear", "bicubic",
                                            "area", "nearest-exact"], value="bicubic", label="Downsampling Method")
                    ds_size = gr.Slider(2, 256, value=32, label="Downsampled Size NxN",
                                        info="Downsamples the image to size NxN if it is larger than NxN")
                with gr.Row():
                    opt = gr.Dropdown(["Adam", "SGD", "Ranger21"], value="Adam", label="Optimizer")
                    lr = gr.Slider(0, 2, value=0.4, label="Learning Rate")
                    steps = gr.Slider(1, 800, value=200, label="Optimization Steps")
                    batch_size = gr.Slider(1, 16, value=1, step=1, label="Batch Size")
                with gr.Row():
                    l2 = gr.Slider(10, 1000, value=100, label="MSE Coefficient")
                    ld = gr.Slider(0.001, 1, value=0.1, label="Geodesic Loss Coefficient")
                with gr.Row():
                    add_noise = gr.Checkbox(label="Add Noise", value=False)
                    noise_stddev = gr.Slider(0, 0.333, value=0.1, label="Noise STD Dev")
                with gr.Row():
                    seed = gr.Number(value=0, precision=0, label="Seed", info="If seed is left as 0, it is randomly determined")
                btn = gr.Button("Generate")
            with gr.Column(elem_id="right"):
                with gr.Row():
                    with gr.Column(elem_id="imgs_left"):
                        down = gr.Gallery(label="Downscaled", elem_classes="ds_img").style(preview=True)
                        ref = gr.Image(label="Reference", elem_classes="ds_img").style(height=256, width=256)
                    with gr.Column():
                        up = gr.Gallery(label="Upsampled", elem_classes="ds_img").style(preview=True)
                loss_text = gr.Textbox(label="Loss")
        btn.click(fn=generate, inputs=[inp_img, opt, ds_method, ds_size, lr, steps, batch_size, l2, ld, noise_stddev, seed, add_noise], outputs=[ref, down, up, loss_text])
    with gr.Tab("Draw Noise"):
        with gr.Row():
            with gr.Column(elem_id="left"):
                inp_img = gr.Image(tool="sketch", shape=None)
                with gr.Row():
                    ds_method = gr.Dropdown(["nearest", "bilinear", "bicubic",
                                            "area", "nearest-exact"], value="bicubic", label="Downsampling Method")
                    ds_size = gr.Slider(2, 256, value=32, label="Downsampled Size NxN",
                                        info="Downsamples the image to size NxN if it is larger than NxN")
                with gr.Row():
                    opt = gr.Dropdown(["Adam", "SGD", "Ranger21"], value="Adam", label="Optimizer")
                    lr = gr.Slider(0, 2, value=0.4, label="Learning Rate")
                    steps = gr.Slider(1, 800, value=200, label="Optimization Steps")
                    batch_size = gr.Slider(1, 16, value=1, step=1, label="Batch Size")
                with gr.Row():
                    l2 = gr.Slider(10, 1000, value=100, label="MSE Coefficient")
                    ld = gr.Slider(0.001, 1, value=0.1, label="Geodesic Loss Coefficient")
                with gr.Row():
                    noise_stddev = gr.Slider(0, 0.333, value=0.1, label="Noise STD Dev")
                with gr.Row():
                    seed = gr.Number(value=0, precision=0, label="Seed", info="If seed is left as 0, it is randomly determined")
                btn = gr.Button("Generate")
            with gr.Column(elem_id="right"):
                with gr.Row():
                    with gr.Column(elem_id="imgs_left"):
                        down = gr.Gallery(label="Downscaled", elem_classes="ds_img").style(preview=True)
                        ref = gr.Image(label="Reference", elem_classes="ds_img").style(height=256, width=256)
                    with gr.Column():
                        up = gr.Gallery(label="Upsampled", elem_classes="ds_img").style(preview=True)
                loss_text = gr.Textbox(label="Loss")
        btn.click(fn=generate, inputs=[inp_img, opt, ds_method, ds_size, lr, steps, batch_size, l2, ld, noise_stddev, seed], outputs=[ref, down, up, loss_text])

app.launch()