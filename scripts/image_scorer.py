import copy
from hashlib import md5
import json
import os
from modules import sd_samplers, shared, scripts, script_callbacks
from modules.script_callbacks import ImageSaveParams
import modules.images as images
from modules.processing import Processed, process_images, StableDiffusionProcessing
from modules.shared import opts, OptionInfo
from modules.paths import script_path

import gradio as gr
from pathlib import Path
import torch
import torch.nn as nn
import clip
import platform
from launch import is_installed, run_pip
from modules.generation_parameters_copypaste import parse_generation_parameters

extension_name = "Aesthetic Image Scorer"
if platform.system() == "Windows" and not is_installed("pywin32"):
    run_pip(f"install pywin32", "pywin32")
try:
    from tools.add_tags import tag_files
except:
    print(f"{extension_name}: Unable to load Windows tagging script from tools directory")
    tag_files = None

state_name = "sac+logos+ava1-l14-linearMSE.pth"
if not Path(state_name).exists():
    url = f"https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/{state_name}?raw=true"
    import requests
    r = requests.get(url)
    with open(state_name, "wb") as f:
        f.write(r.content)


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


try:
    force_cpu = opts.ais_force_cpu
except:
    force_cpu = False

if force_cpu:
    print(f"{extension_name}: Forcing prediction model to run on CPU")
device = "cuda" if not force_cpu and torch.cuda.is_available() else "cpu"
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

class AISGroup:
    def __init__(self, name="", apply_choices=lambda choice, choice_values: {"tags": [f"{choice}_{choice_values[choice]}"]}, default=[]):
        gen_params = lambda p: {
            "steps": p.steps,
            "sampler": sd_samplers.samplers[p.sampler_index].name,
            "cfg_scale": p.cfg_scale,
            "seed": p.seed,
            "width": p.width,
            "height": p.height,
            "model": shared.sd_model.sd_model_hash,
            "prompt": p.prompt,
            "negative_prompt": p.negative_prompt,
        }
        self.choice_processors = {
            "aesthetic_score": lambda params: params.pnginfo["aesthetic_score"] if "aesthetic_score" in params.pnginfo else round(get_score(params.image), 1),
            "sampler": lambda params: sd_samplers.samplers[params.p.sampler_index].name if params.p is not None and params.p.sampler else None,
            "cfg_scale": lambda params: params.p.cfg_scale if params.p is not None and params.p.cfg_scale else None,
            "sd_model_hash": lambda params: shared.sd_model.sd_model_hash,
            "seed": lambda params: str(int(params.p.seed)) if params.p is not None and params.p.seed else None,
            "hash": lambda params: md5(json.dumps(gen_params(params.p)).encode('utf-8')).hexdigest() if params.p is not None else None,
        }
        self.name = name
        self.apply_choices = apply_choices
        self.default = default

    def get_choices(self):
        return list(self.choice_processors.keys())

    def get_name(self):
        return self.name

    def selected(self, opts):
        return opts.__getattr__(self.name)

    def get_choice_processors(self):
        return self.choice_processors

    def apply(self, choice_values, applied, opts):
        for choice in self.get_choices():
            if choice in self.selected(opts) and choice in choice_values:
                applied_choices = self.apply_choices(choice, choice_values)
                for key, value in applied_choices.items():
                    if key not in applied:
                        applied[key] = self.get_default()
                    if isinstance(applied[key], dict):
                        applied[key].update(value)
                    else:
                        applied[key] = applied[key] + value
        return applied
    
    def get_default(self):
        return self.default 

class AISGroups:
    def __init__(self, groups=[]):
        self.groups = groups
        self.choices = {}
        self.choice_processors = {}
        for group in groups:
            self.choice_processors.update(group.get_choice_processors())
        self.choices = list(self.choice_processors.keys())

    def all_selected(self, opts):
        selected = {}
        for group in self.groups:
            for select in group.selected(opts):
                selected.update({select: 0})
        return selected.keys()

    def apply(self, opts: list, params: ImageSaveParams):
        parsed_info = parse_generation_parameters(
            params.pnginfo.get("parameters", ""))
        params_hash = md5(json.dumps(
            str(vars(params)), default=lambda o: dir(o), sort_keys=True).encode('utf-8')).hexdigest()
        applied = {}
        if params_hash not in output_cache:
            choice_values = {}
            choices_selected = self.all_selected(opts)
            for choice, processor in self.choice_processors.items():
                if choice in choices_selected:
                    choice_values.update({choice: processor(params)})

            if "seed" in choice_values and int(choice_values["seed"]) == -1:
                choice_values["seed"] = int(parsed_info["Seed"]) if "Seed" in parsed_info else int(choice_values["seed"])
            
            for group in self.groups:
                applied = group.apply(choice_values, applied, opts)

            expected_keys = ["tags", "categories", "info", "pnginfo"]
            for key in expected_keys:
                if key not in applied:
                    applied[key] = []
            
            output_cache[params_hash] = applied
        else:
            applied = output_cache[params_hash]
            output_cache.clear()
        
        return applied


ais_exif_pnginfo_choices = AISGroup(name="ais_exif_pnginfo_group", apply_choices=lambda choice, choice_values: {
                                              "pnginfo": {choice: choice_values[choice]}}, default={})
ais_windows_tag_group_choices = AISGroup(name="ais_windows_tag_group")
ais_windows_category_group_choices = AISGroup(name="ais_windows_category_group", apply_choices=lambda choice, choice_values: {
                                              "categories": [f"{choice}_{choice_values[choice]}"]})
ais_generation_params_text_choices = AISGroup(name="ais_generation_params_text_group", apply_choices=lambda choice, choice_values: {
                                              "info": {choice: choice_values[choice]}}, default={})

ais_group = AISGroups([
    ais_windows_tag_group_choices,
    ais_windows_category_group_choices,
    ais_generation_params_text_choices,
    ais_exif_pnginfo_choices,
])

output_cache = {}

def on_ui_settings():
    options = {}

    options.update(shared.options_section(('ais', extension_name), {
        ais_exif_pnginfo_choices.get_name(): OptionInfo([], "Save score as EXIF or PNG Info Chunk", gr.CheckboxGroup, {"choices": ais_exif_pnginfo_choices.get_choices()}),
        ais_windows_tag_group_choices.get_name(): OptionInfo([], "Save tags (Windows only)", gr.CheckboxGroup, {"choices": ais_windows_tag_group_choices.get_choices()}),
        ais_windows_category_group_choices.get_name(): OptionInfo([], "Save category (Windows only)", gr.CheckboxGroup, {"choices": ais_windows_category_group_choices.get_choices()}),
        ais_generation_params_text_choices.get_name(): OptionInfo([], "Save generation params text", gr.CheckboxGroup, {"choices": ais_generation_params_text_choices.get_choices()}),
        "ais_force_cpu": OptionInfo(False, "Force CPU (Requires Custom Script Reload)"),
    }))

    opts.add_option(ais_exif_pnginfo_choices.get_name(),
                    options[ais_exif_pnginfo_choices.get_name()])
    opts.add_option(ais_windows_tag_group_choices.get_name(),
                    options[ais_windows_tag_group_choices.get_name()])
    opts.add_option(ais_windows_category_group_choices.get_name(),
                    options[ais_windows_category_group_choices.get_name()])
    opts.add_option(ais_generation_params_text_choices.get_name(),
                    options[ais_generation_params_text_choices.get_name()])
    opts.add_option("ais_force_cpu", options["ais_force_cpu"])

def on_before_image_saved(params: ImageSaveParams):    
    applied = ais_group.apply(opts, params)
    if len(applied["pnginfo"]) > 0:
        params.pnginfo.update(applied["pnginfo"])
    
    if len(applied["info"]) > 0:
        parts = []
        for label, value in applied["info"].items():
            parts.append(f"{label}: {value}")
        if len(parts) > 0:
            if len(params.pnginfo["parameters"]) > 0:
                params.pnginfo["parameters"] += ", "
            params.pnginfo["parameters"] += f"{', '.join(parts)}\n"
    
    return params
        
def on_image_saved(params: ImageSaveParams):
    filename = os.path.realpath(os.path.join(script_path, params.filename))
    applied = ais_group.apply(opts, params)
    if tag_files is not None:
        tag_files(filename=filename, tags=applied["tags"], categories=applied["categories"],
                    log_prefix=f"{extension_name}: ")
    else:
        print(f"{extension_name}: Unable to load tagging script")

class AestheticImageScorer(scripts.Script):
    def title(self):
        return extension_name

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def process(self, p):
        pass


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_before_image_saved(on_before_image_saved)
script_callbacks.on_image_saved(on_image_saved)
