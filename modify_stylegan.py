import pickle
import os
import numpy as np
os.sys.path.append(os.getcwd() +'/stylegan3')
import modified_stylegan2

model_name = "stylegan2-ffhq-256x256.pkl"
with open(model_name, 'rb') as f:
    model = pickle.load(f)
G = model['G_ema'].cuda()

modified = modified_stylegan2.Generator(G.z_dim, G.c_dim, G.w_dim, G.img_resolution, G.img_channels, channel_base=16384)

for name, module in modified.named_parameters():
    if name in G.state_dict():
        module.data = G.state_dict()[name].data

with open("stylegan2-ffhq-256x256.pkl", 'wb') as f:
    data = dict(G=model['G'], D=model['D'] ,G_ema=modified,
                training_set_kwargs=model['training_set_kwargs'],
                augment_pipe=model['augment_pipe'], kwargs=model['kwargs'])
    pickle.dump(data, f)