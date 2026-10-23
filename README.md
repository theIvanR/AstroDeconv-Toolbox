# Astro Deconvolution Toolbox

A modular MATLAB pipeline for astrophotography preprocessing and deconvolution. The toolbox prepares light frames for stacking, then sharpens the resulting image using an empirically estimated point-spread function (PSF) derived from stars in the image.

The workflow is designed to keep calibration, stacking, and deconvolution as separate stages, allowing you to use your preferred stacking software while retaining control over the image restoration process.

## Features

- **Frame calibration:** Prepare light frames using calibration data such as darks and flats.
- **Flexible input workflow:** Organize light, flat, and dark frames in dedicated directories.
- **RAW conversion support:** Convert camera RAW files using the provided third-party tools, where required.
- **Empirical PSF estimation:** Detect stars, extract star patches, and estimate a PSF from the observed image.
- **Per-channel deconvolution:** Process an individual image channel or all available channels independently.
- **Richardson–Lucy restoration:** Apply iterative deconvolution using the estimated PSF.
- **Modular MATLAB implementation:** Individual processing stages are separated into reusable functions.

## Requirements

- MATLAB
- Image Processing Toolbox for image-processing functions such as `imread`, `imwrite`, `im2double`, and `deconvlucy`
- Third-party RAW conversion tools, if your input files require conversion

Exact requirements may depend on the calibration functions and conversion tools included in your installation.

## Project Structure

```text
Astro-Deconv-Toolbox/
├── calibrate_frames.m
├── deconv.m
├── functions/
├── input/
│   ├── light/
│   ├── flat/
│   └── dark/
├── output/
├── _gen_synth_data/
└── _third party/
```

The `input/` and `output/` directories are created automatically when needed. The `third party/` directory contains optional tools for converting RAW frames to a supported image format.

*The function filenames above illustrate the current pipeline; adjust the list to match the actual repository contents.*

## Workflow

### 1. Prepare your input frames

Place your source frames in the corresponding directories:

- `input/light/` — Light frames containing your astronomical target.
- `input/flat/` — Flat frames used to correct illumination variations and optical vignetting.
- `input/dark/` — Dark frames used to correct dark current and related sensor effects.

Keep your original files unchanged so that you can return to the raw data if necessary.

### 2. Convert RAW frames if necessary

If your camera produces RAW files that are not directly supported by the calibration pipeline, use the conversion tools in `third party/` to convert them into a supported format, such as TIFF.

Where possible, preserve linear intensity data and avoid applying gamma correction, sharpening, denoising, or automatic contrast adjustments during conversion.

### 3. Calibrate the frames

Run the MATLAB script:

```matlab
calibrate_frames
```

This stage prepares the light frames using the available calibration data and writes the resulting frames to `output/`.

The calibrated frames are intended for subsequent alignment and stacking.

### 4. Stack the calibrated frames

Use the stacking tool of your choice to align and combine the prepared frames.

Stacking is deliberately kept separate from this toolbox, allowing you to choose the alignment, rejection, and integration methods that best suit your data.

Save the resulting stacked image as a supported image format, such as TIFF, in the expected output location for the deconvolution script.

### 5. Deconvolve the stacked image

Run:

```matlab
deconv
```

The deconvolution pipeline:

1. Loads the stacked image.
2. Applies background neutralization.
3. Selects the requested image channel or processes all channels.
4. Detects suitable stars in each selected channel.
5. Extracts and normalizes star patches.
6. Constructs an empirical PSF.
7. Applies Richardson–Lucy deconvolution.
8. Reassembles the processed channels and saves the resulting image.

The PSF is estimated independently for each processed channel, allowing the restoration to account for differences in the observed stellar profiles between channels.

## Configuration

The deconvolution script exposes parameters for selecting the channel and controlling the restoration.

For example:

```matlab
cfg.channel = 0;       % 0 = all channels
                       % 1 = red, 2 = green, 3 = blue
cfg.iterations = 3;    % Richardson–Lucy iterations
```

The current default of three iterations is an empirical starting point for the images tested so far, rather than a universal optimum.

Other configurable parameters control star selection, patch size, minimum star separation, and saturation rejection. These may need adjustment for different sensors, exposure levels, telescopes, and sky conditions.

## Enoy
