# upsampling
## How to run:
1. Clone the repository with the StyleGAN3 submodule
```
git clone --recursive https://github.com/eunalan/upsampling.git
```
2. Download the stylegan2-ffhq-256x256.pkl pretrained model from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan2, and place it in the cloned directory.

3. Install the required Python packages
```
pip install -r requirements.txt
```

4. Run the application with Gradio, by default it starts on http://127.0.0.1:7860
```
gradio main.py
```
