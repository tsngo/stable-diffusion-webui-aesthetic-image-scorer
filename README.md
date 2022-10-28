# Aesthetic Image Scorer

![](tag_group_by.png)

Extension for https://github.com/AUTOMATIC1111/stable-diffusion-webui

Calculates aestetic score for generated images using [CLIP+MLP Aesthetic Score Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) based on [Chad Scorer](https://github.com/grexzen/SD-Chad/blob/main/chad_scorer.py)

See [Discussion](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/1831)

Saves score to windows tags with other options planned

## Installation
Clone the repo into the `extensions` directory and restart the web ui:

```commandline
git clone --recurse-submodules https://github.com/tsngo/stable-diffusion-webui-aesthetic-image-scorer extensions/aesthetic-image-scorer
```

To upgrade do:

```commandline
git pull
```
After restarting the ui, see settings for options

## Features
- Save score as EXIF or PNG Info Chunk (Only PNG Chunk Info currently)
- Save score as tag (Windows Only)
    - Added to tags as `aesthetic_score_5.9`
    - JPG supports by default. PNG tags requires a 3rd party software like [File Metadata](https://github.com/Dijji/FileMeta/releases)
- Run prediction using CPU

## FAQ
- There is a bug currently preventing writing to PNG Info Chuck
    - See [AUTOMATIC1111/stable-diffusion-webui#3723](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/3723)
- If seeing this error `Aesthetic Image Scorer: Unable to write tag or category`
    - probably related to above bug but the windows tagging should still work
- If seeing this error `Aesthetic Image Scorer: Unable to load Windows tagging script from tools directory`
    - The tools submodule didn't get pulled. Make sure to use follow the installation instructions
