# Aesthetic Image Scorer

![](tag_group_by.png)

Extension for https://github.com/AUTOMATIC1111/stable-diffusion-webui

Calculates aestetic score for generated images using [CLIP+MLP Aesthetic Score Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) based on [Chad Scorer](https://github.com/grexzen/SD-Chad/blob/main/chad_scorer.py)

See [Discussion](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/1831)

Saves score to windows tags with other options planned

## Installation
Clone the repo into the `extensions` directory and restart the web ui:

```commandline
git clone https://github.com/tsngo/stable-diffusion-webui-aesthetic-image-scorer extensions/aesthetic-image-scorer
```

or use the `Extensions` tab of the webui to `Install from URL`

```commandline
https://github.com/tsngo/stable-diffusion-webui-aesthetic-image-scorer
```


To upgrade do:

```commandline
git pull
```

or use `Extensions` tab to upgrade.

After restarting the ui, see settings for options

## Features
![](settings_section.png)
- Save aesthetic score and other things
    - `hash` is a md5 hash of prompt, negative prompt, dimensions, steps, cfg scale, seed, model hash (experimental). Don't rely on this. 
    - rest should be obvious
- Save as EXIF or PNG Info Chunk (Only PNG Chunk Info currently)
- Save as tag (Windows Only)
    - Added to tags as `aesthetic_score_5.9`
    - JPG supports by default. PNG tags requires a 3rd party software like [File Metadata](https://github.com/Dijji/FileMeta/releases)
- Save as category (Windows Only)
- Save as generation parameter text
- Run prediction using CPU

## FAQ
- There is a bug currently preventing writing to PNG Info Chuck
    - See [AUTOMATIC1111/stable-diffusion-webui#3723](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/3723)
- If seeing this error `Aesthetic Image Scorer: Unable to write tag or category`
    - probably related to above bug but the windows tagging should still work
- If seeing this error `Aesthetic Image Scorer: Unable to load Windows tagging script from tools directory`
    - check if the tools directory has files. Submodules no longer required.
