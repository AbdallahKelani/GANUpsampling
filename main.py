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

# Schedulers from https://github.com/adamian98/pulse
schedule_dict = {
            'fixed': lambda x: 1,
            'linear1cycle': lambda x: (9*(1-np.abs(x/steps-1/2)*2)+1)/10,
            'linear1cycledrop': lambda x: (9*(1-np.abs(x/(0.9*steps)-1/2)*2)+1)/10 if x < 0.9*steps else 1/10 + (x-0.9*steps)/(0.1*steps)*(1/1000-1/10),
        }

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

def downsample(img, size=(32, 32)):
    img_32 = torch.nn.functional.interpolate(img, size=size, mode='bicubic')[0]
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
    print(img.shape)
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

def genereate_stylegan_noise(mean, stddev, mask=None):
    noises = []
    for res in [4, 8, 16, 32, 64, 128, 256]:
        noise = np.random.normal(mean, stddev, [1,1,res,res]).astype(np.float32)
        noise = torch.from_numpy(noise).cuda()

        if mask is not None:
            m = np.float32(np.all(mask[:, :, :3] == 255, axis=2)) * 255
            m = np.expand_dims(m, axis=0)
            m = np.expand_dims(m, axis=0)
            m = torch.tensor(m).cuda()
            m = downsample(m, size=(res,res))
            noise = (noise * m).detach()
        noises.append(noise)
    noises[6] = torch.zeros([1,1,256,256]).cuda().detach()
    return noises


def generate(img, optimizer, ds_method, ds_size, lr, steps, 
             batch_size, c_l2, c_ld, c_pairwise, ref_noise_stddev, stylegan_noise_stddev,
             seed, use_mapping_net, different_noise_same_latent, add_ref_noise, add_stylegan_noise):
    #different_noise_same_latent = True
    use_scheduler = (optimizer == torch.optim.Adam or optimizer == torch.optim.SGD)

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

    ref = img.astype(np.float32)
    ref = np.expand_dims(ref, axis=0)
    ref = np.transpose(ref, (0, 3, 1, 2))
    ref = torch.tensor(ref).cuda()
    ref_32 = torch.nn.functional.interpolate(ref, size=ds_size, mode=ds_method)
    ref_32 = [torch.clamp(ref_32 / 255, 0, 1)]
    print(add_ref_noise, different_noise_same_latent, "ASDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
    if add_ref_noise:
        ref_32 = ref_32[0]
        if different_noise_same_latent:
            ref_imgs = []
            for _ in range(batch_size):
                print("Here")
                ref_imgs.append(add_gaussian_noise(ref_32, 0, ref_noise_stddev, mask).detach())
            ref_32 = ref_imgs
        else:
            ref_32 = [add_gaussian_noise(ref_32, 0, ref_noise_stddev, mask).detach()]
    
    #ref_32 = ref_32.detach()
    if add_ref_noise and different_noise_same_latent:
        tmp_bs = batch_size
        batch_size = 1

    if use_mapping_net:
        z = torch.from_numpy(np.random.randn(batch_size, G.z_dim)).cuda()
        latent = G.mapping(z, None, truncation_psi=0.5, truncation_cutoff=8)
        latent.requires_grad = True
    else:
        latent = torch.randn((batch_size, 14, 512), dtype=torch.float, requires_grad=True, device='cuda')
    
    if add_ref_noise and different_noise_same_latent:
        batch_size = tmp_bs
        latent = latent.detach()
        latent = torch.tensor(latent.tile((batch_size, 1, 1)), requires_grad=True, device='cuda')

    opt_settings = {"lr": lr}
    if optimizer == "Ranger21":
        opt_settings["num_epochs"] = steps
        opt_settings["num_batches_per_epoch"] = batch_size
    optim = SphericalOptimizer(optimizers[optimizer], [latent], **opt_settings)

    if use_scheduler:
        lr_schedule = 'linear1cycledrop'
        schedule_func = schedule_dict[lr_schedule]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim.opt, schedule_func)

    noise_mode = 'random'
    custom_noise = None
    if add_stylegan_noise:
        custom_noise = genereate_stylegan_noise(0, stylegan_noise_stddev, mask)
        noise_mode = 'custom'
        best_latents = [None] * batch_size
    min_loss = [np.inf] * batch_size
    best = [None] * batch_size
    best_32 = [None] * batch_size
    for i in range(steps):
        optim.opt.zero_grad()
        images = []
        batch_loss = 0
        for b in range(batch_size):
            w = leaky(latent[None, b] * gaussian_fit["std"] + gaussian_fit["mean"])
            #img = (G.synthesis(w, noise_mode='random', force_fp32=True) + 1) / 2 # NCHW, float32, dynamic range [-1, +1], no truncation
            img = (G.synthesis(w, noise_mode=noise_mode, custom_noise=custom_noise, force_fp32=True) + 1) / 2 # NCHW, float32, dynamic range [-1, +1], no truncation
            #img = (G.synthesis(w, noise_mode='none', force_fp32=True) + 1) / 2 # NCHW, float32, dynamic range [-1, +1], no truncation
            img = img.clamp(0, 1)
            img_32 = torch.nn.functional.interpolate(img, size=ds_size, mode=ds_method)
            img_32 = img_32.clamp(0, 1)
            if add_ref_noise and different_noise_same_latent:
                l = loss(w, img_32, ref_32[b], c_l2, c_ld)
            else:
                l = loss(w, img_32, ref_32[0], c_l2, c_ld)
            print(l, end='\r')
            if (l < min_loss[b]):
                with torch.no_grad():
                    min_loss[b] = l.detach()
                    best[b] = img.clone().detach()
                    best_32[b] = img_32.clone().detach()
                    if add_stylegan_noise:
                        best_latents[b] = w
            batch_loss += l
            #l.backward()
            if c_pairwise != 0:
                images.append(img)
        if c_pairwise != 0:
            pairwise_loss = 0
            images = torch.stack(images)
            for i in range(batch_size):
                for j in range(i + 1, batch_size):
                    pairwise_loss += torch.dist(images[i], images[j], 2)
            batch_loss -= (pairwise_loss / batch_size) * c_pairwise
        batch_loss.backward()
        optim.step()
        if use_scheduler:
            scheduler.step()

    if add_stylegan_noise:
        for b in range(batch_size):
            best[b] = ((G.synthesis(best_latents[b], noise_mode='random', force_fp32=True) + 1) / 2).detach()
    

    ref_32 = [x.cpu().numpy().transpose((0, 2, 3, 1))[0] for x in ref_32]
    best_32 = [x.cpu().numpy().transpose((0, 2, 3, 1))[0] for x in best_32]
    best = [x.cpu().numpy().transpose((0, 2, 3, 1))[0] for x in best]

    return ref_32, \
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
                    opt = gr.Dropdown(["Adam", "SGD", "Ranger21"], value="Ranger21", label="Optimizer")
                    lr = gr.Slider(0, 2, value=0.4, label="Learning Rate")
                    steps = gr.Slider(1, 800, value=200, label="Optimization Steps")
                    batch_size = gr.Slider(1, 16, value=1, step=1, label="Batch Size")
                with gr.Row():
                    l2 = gr.Slider(10, 1000, value=405, label="MSE Coefficient") # 100
                    ld = gr.Slider(0.001, 1, value=0.058, label="Geodesic Loss Coefficient") # 0.1
                    l_pairwise = gr.Slider(0.0, 0.1, value=0, label="Pairwise Batch Loss Coefficient")
                with gr.Row():
                    add_ref_noise = gr.Checkbox(label="Add Img Noise", value=False, info="Gaussian noise with mean 0")
                    ref_noise_stddev = gr.Slider(0, 0.333, value=0.1, label="Image Noise STD Dev")
                    add_stylegan_noise = gr.Checkbox(label="Add StyleGAN Noise", value=False, info="Gaussian noise with mean 0")
                    stylegan_noise_stddev = gr.Slider(0, 0.333, value=0.1, label="StyleGAN Noise STD Dev")
                    different_noise_same_latent = gr.Checkbox(label="Use different noise on same latents", value=False
                                                              , info="Samples different noise to add to reference images, but keeps the latents same across a batch")
                with gr.Row():
                    seed = gr.Number(value=0, precision=0, label="Seed", info="If seed is left as 0, it is randomly determined")
                    use_mapping_net = gr.Checkbox(label="Use mapping network", value=False
                                            , info="Uses the StyleGAN mapping network to produce 14x512 latent vectors from a single 512 vector.\
                                                    All 14 latent vectors are randomly initialized if left unchecked.")
                btn = gr.Button("Generate")
            with gr.Column(elem_id="right"):
                with gr.Row():
                    with gr.Column(elem_id="imgs_left"):
                        down = gr.Gallery(label="Downscaled", elem_classes="ds_img").style(preview=True)
                        ref = gr.Gallery(label="Reference", elem_classes="ds_img").style(preview=True)
                        #ref = gr.Image(label="Reference", elem_classes="ds_img").style(height=256, width=256)
                    with gr.Column():
                        up = gr.Gallery(label="Upsampled", elem_classes="ds_img").style(preview=True)
                loss_text = gr.Textbox(label="Loss")
        btn.click(fn=generate, inputs=[inp_img, opt, ds_method, ds_size, lr, steps, batch_size, l2, ld, l_pairwise, ref_noise_stddev, stylegan_noise_stddev, seed, use_mapping_net, different_noise_same_latent, add_ref_noise, add_stylegan_noise], outputs=[ref, down, up, loss_text])
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
                    opt = gr.Dropdown(["Adam", "SGD", "Ranger21"], value="Ranger21", label="Optimizer")
                    lr = gr.Slider(0, 2, value=0.4, label="Learning Rate")
                    steps = gr.Slider(1, 800, value=200, label="Optimization Steps")
                    batch_size = gr.Slider(1, 16, value=1, step=1, label="Batch Size")
                with gr.Row():
                    l2 = gr.Slider(10, 1000, value=405, label="MSE Coefficient") # 100
                    ld = gr.Slider(0.001, 1, value=0.058, label="Geodesic Loss Coefficient") # 0.1  
                    l_pairwise = gr.Slider(0.0, 0.1, value=0, label="Pairwise Batch Loss Coefficient")
                with gr.Row():
                    add_ref_noise = gr.Checkbox(label="Add Img Noise", value=True, info="Gaussian noise with mean 0")
                    ref_noise_stddev = gr.Slider(0, 0.333, value=0.1, label="Image Noise STD Dev")
                    add_stylegan_noise = gr.Checkbox(label="Add StyleGAN Noise", value=False, info="Gaussian noise with mean 0")
                    stylegan_noise_stddev = gr.Slider(0, 0.333, value=0.1, label="StyleGAN Noise STD Dev")
                    different_noise_same_latent = gr.Checkbox(label="Use different noise on same latents", value=False
                                            , info="Samples different noise to add to reference images, but keeps the latents same across a batch")
                with gr.Row():
                    seed = gr.Number(value=0, precision=0, label="Seed", info="If seed is left as 0, it is randomly determined")
                    use_mapping_net = gr.Checkbox(label="Use mapping network", value=False
                                            , info="Uses the StyleGAN mapping network to produce 14x512 latent vectors from a single 512 vector.\
                                                    All 14 latent vectors are randomly initialized if left unchecked.")
                btn = gr.Button("Generate")
            with gr.Column(elem_id="right"):
                with gr.Row():
                    with gr.Column(elem_id="imgs_left"):
                        down = gr.Gallery(label="Downscaled", elem_classes="ds_img").style(preview=True)
                        ref = gr.Gallery(label="Reference", elem_classes="ds_img").style(preview=True)
                        #ref = gr.Image(label="Reference", elem_classes="ds_img").style(height=256, width=256)
                    with gr.Column():
                        up = gr.Gallery(label="Upsampled", elem_classes="ds_img").style(preview=True)
                loss_text = gr.Textbox(label="Loss")
        btn.click(fn=generate, inputs=[inp_img, opt, ds_method, ds_size, lr, steps, batch_size, l2, ld, l_pairwise, ref_noise_stddev, stylegan_noise_stddev, seed, use_mapping_net, different_noise_same_latent, add_ref_noise, add_stylegan_noise], outputs=[ref, down, up, loss_text])

app.launch()