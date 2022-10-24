from modules import sd_samplers, shared, scripts, script_callbacks
import modules.images as images
from modules.processing import Processed, process_images, StableDiffusionProcessing
from modules.shared import opts, OptionInfo

from pathlib import Path
import torch
import torch.nn as nn
import clip
import platform
from launch import is_installed, run_pip

if platform.system() == "Windows" and not is_installed("pywin32"):
    run_pip(f"install pywin32", "pywin32")
try:
    from tools.add_tags import tag_files
except:
    print("Unable to load")
    tag_files = None

state_name = "sac+logos+ava1-l14-linearMSE.pth"
if not Path(state_name).exists():
    url = f"https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/{state_name}?raw=true"
    import requests
    r = requests.get(url)
    with open(state_name, "wb") as f:
        f.write(r.content)

class AestheticImageScorer:
    def __init__(self):
        self.ais_windows_tag = False

    def set_params(self, p, ais_windows_tag=False):
        self.ais_windows_tag = ais_windows_tag

        p.extra_generation_params.update({
            "AIS Windows Tag": ais_windows_tag,
        })

ais = AestheticImageScorer()

class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
# load the model you trained previously or the model available in this repo
pt_state = torch.load(state_name, map_location=torch.device(device=device)) 

# CLIP embedding dim is 768 for CLIP ViT L 14
predictor = AestheticPredictor(768)
predictor.load_state_dict(pt_state)
predictor.to(device)
predictor.eval()

clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

def get_image_features(image, device=device, model=clip_model, preprocess=clip_preprocess):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        # l2 normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().detach().numpy()
    return image_features

def get_score(image):
    image_features = get_image_features(image)
    score = predictor(torch.from_numpy(image_features).to(device).float())
    return score.item()


def on_ui_settings():
    options = {}
    options.update(shared.options_section(('ais', "Aesthetic Image Scorer"), {
        "ais_windows_tag": OptionInfo(False, "Save score as tag (Windows Only)"),
    }))
    opts.add_option("ais_windows_tag", options["ais_windows_tag"])


def on_save_imaged(image, p, fullfn, txt_fullfn):
    score = round(get_score(image), 1)
    if opts.ais_windows_tag:
        if tag_files is not None:
            tags = [f"aesthetic_score_{score}"]
            tag_files(filename=fullfn, tags=tags)
        else:
            print("Unable to load windows tagging script")

class AestheticImageScorer(scripts.Script):
    def title(self):
        return "Aesthetic Image Scorer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def process(self, p):
        ais.set_params(p, bool(opts.ais_windows_tag))

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_save_imaged(on_save_imaged)
