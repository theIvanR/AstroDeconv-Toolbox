
# AstroDeconv Toolbox
**Planet Deconvolver** & **Deep Sky Deconvolver** — preprocessing + deconvolution tools for astronomical imaging

Two MATLAB preprocessing/deconvolution scripts intended to maximize SNR and extract as much detail as possible from astronomical images:

- `deepsky_deconvolving_preprocessor.m` — tuned for faint, extended targets (nebulae, galaxies).
- `planetary_deconvolving_preprocessor.m` — tuned for bright, high-SNR small-disk targets (planets, Moon, bright satellites).

> Why two tools? SNR and the dominant error modes differ between planetary and deep-sky imaging. Planet code assumes high signal, compact PSF, and benefits most from per-channel Wiener deconvolution. Deep-sky code focuses on low SNR, Poisson noise, careful regularization and PSF confidence weighting.

---

## Overview
These scripts form a small toolbox that:

- Converts RAW (or pre-stacked) images to linear RGB when needed.
- Builds robust master flats and confidence-weighted inverse flats (deep-sky pipeline).
- Applies robust per-channel or luminance-based deconvolution (planetary pipeline).
- Contains hooks for hot-pixel removal, auto white balance, gaussian low-pass filtering, CNN denoising (optional), PSF estimation from nearby star patches, and (semi) blind deconvolution.
- Saves high-dynamic-range results as FP32 TIFFs (or clipped uint16 if desired).

The goal is to **maximize final SNR by improving data quality** (better flat correction, PSF estimation, denoising, and tailored deconvolution) — because SNR scales linearly with raw signal quality, but only with the square root of the number of frames.

---

## Features
- Robust master-flat creation with SNR-based confidence maps and radial tapering.
- Hot pixel removal (Hampel/neighbour average).
- Auto white balance using bright-pixel robust medians.
- PSF estimation from star centroids + median patch stack.
- Per-channel Wiener deconvolution for planetary work.
- (Semi-)blind deconvolution step (optional) for PSF refinement.
- Optional DnCNN denoising (requires Deep Learning Toolbox).
- Parallel processing friendly (uses `parfor` in the light-frame loop if MATLAB Parallel Toolbox is available).
- Diagnostic output folder for visualizing master flat, inverse, and confidence maps.

---

## Requirements
- MATLAB R2020a or later (tested with R2022a+ features)
- Image Processing Toolbox
- (Optional) Deep Learning Toolbox — for `denoisingNetwork("DnCNN")`
- `rawread` or other RAW reader if you plan to convert `.ARW` inside the pipeline
- (Optional) Parallel Computing Toolbox for `parfor` speedups

---

## Repository layout (suggested)
```
/repo-root
  README.md
  deepsky_deconvolving_preprocessor.m
  planetary_deconvolving_preprocessor.m
  /Input          % put RAW or stacked light frames here
  /Flat           % optional flat frames
  /Output         % results will be written here
  /flat_diag      % optional diagnostics (created when enabled)
```

---

## Quick start

1. Clone this repo and open MATLAB.
2. Put your **light frames** into `Input/`. Put **flat frames** (if available) into `Flat/`.
3. Run the preprocessor of interest:

```matlab
% Deep-sky pipeline (prepares master flats, performs preprocessing + deconv)
run('deepsky_deconvolving_preprocessor.m')

% Planetary pipeline (expects a stacked planet image and a nearby star PSF)
run('planetary_deconvolving_preprocessor.m')
```

4. Output TIFFs are written to `Output/` (FP32 scaled by default). See code-head comments for `saveMode` options.

---

## Usage & Examples

### Deep Sky (recommended workflow)
1. Preprocess your RAW frames or use pre-stacked frames (calibrated).
2. Put flat frames in `Flat/` (recommended). The deep-sky preprocessor builds a `masterFlat` and `masterFlatInv` using a robust percentile normalization and SNR-based confidence weighting.
3. The script will:
   - Create master flat if flats available.
   - Apply flat correction to each light frame.
   - Perform optional denoise & low-pass filtering.
   - Estimate PSF (if bright stars available) and perform (semi) blind deconvolution or regularized deconvolution tuned for faint structures.
4. Outputs: `Output/*.tif` and optional diagnostics in `flat_diag/`.

### Planetary (recommended workflow)
1. Do your usual multi-frame alignment & stacking in your favorite stacking program (Autostakkert!, RegiStax, etc.). Produce:
   - A high-SNR stacked planet image (target).
   - A nearby star or point-source frame for PSF estimation (ideally taken close in time & airmass).
2. Place the stacked planet image as `Input/planet_stack.tif` and the star/PSF image as `Input/psf_star.tif` (or adjust the script paths).
3. Run `planetary_deconvolving_preprocessor.m`. It:
   - Reads `y` (blurry planet) and `h` (PSF/star).
   - Centers & normalizes the PSF energy.
   - Performs **per-channel Wiener** deconvolution (robust to noise with tunable NSR).
4. Output: `Output/planet_deconv.tif` (FP or uint16 as configured).

---

## Key parameters (defaults shown in code)
- `blackLevel` — black level subtraction for RAW→RGB conversion (per CFA channel).
- `useAutoWB` — `true`/`false`, whether to apply automatic white-balance.
- `doHampel` — `true`/`false`, apply hot pixel removal on CFA.
- `params.percentileScale` — percentile used to normalize master flat (default 99).
- `params.minNormVal` — floor value for normalized flat pixels to avoid huge gains.
- `params.maxGain` — hard cap for inverse gains (defaults in code).
- `params.snrPercentileClipping` — percentile for determining robust SNR max.
- `params.doRadialTaper` — enable radial taper blending for low SNR regions.
- Planet pipeline: `NSR` (noise-to-signal ratio for `deconvwnr`) — tune between `1e-1` (aggressive smoothing) down to `1e-4` (very aggressive deconv) depending on noise.

**Suggested starting values**
- Planet: `NSR = 1e-2` → good balance for planetary stacks with decent SNR.
- Deep sky: prefer Lucy–Richardson or regularized deconvolution with fewer iterations (10–50) and denoising beforehand.

---

## Tips & Practical Advice

**Flat frames:** use well-exposed flats (avoid saturated pixels), take many flats and median-stack them. If flats are low SNR at edges, the pipeline's SNR weighting + radial taper helps avoid overcorrection.

**PSF capture for planets:** use a nearby isolated star taken with similar focus/seeing and as close in time/airmass as possible. Recentering the PSF is crucial — your pipeline recenters the PSF centroid before deconvolution.

**Denoising:** if you enable CNN denoising (DnCNN), denoise *before* PSF estimation to avoid biasing the star shape. But be careful — some denoisers can slightly alter PSF fine structure.

**Autoscaling in visualization:** saved FP32 TIFFs are scaled in the pipeline; preview with tools that support 32-bit floats or convert to uint16 for display.

**Avoid over-deconvolution:** ringing is common. Start conservative (small number of iterations or larger NSR) and increase only if results are physically plausible.

---

## Troubleshooting
- `rawread` errors: ensure your MATLAB supports `rawread` or use an external pre-conversion step (e.g., use `dcraw`/`rawtherapee` to export linear TIFFs).
- `No flats found`: the script will create `Flat/` if missing; add `.arw` or linear TIFF flats into it or the master flat step will be skipped.
- PSF estimation fails: lower the brightness threshold or provide a manually cropped PSF image.

---

## API / Function prototypes (quick reference)
These scripts are currently single-file pipelines, but the important helper functions inside are:

```matlab
% Convert RAW to RGB (returns single HxWx3 linear RGB)
rgb = rawToRGB(rawFilePath, 'BlackLevel', [bR bG1 bG2 bB], 'RemoveHotPixels', true);

% Remove hot pixels on CFA image
clean = removeHotPixelsRAW(cfaImage);

% Gaussian low-pass in frequency domain
filtered = applyGaussianLowPass(img, minPixelDetail);

% Estimate PSF from star centroids
[psf_est, centroids, patch_stack] = estimate_star_psf(rgbSignal, patch_size, bright_thresh);

% Smart (semi-blind) deconvolution
[x_recovered, h_est] = smartDeconvRGB(y, h_init, numIter, NSR, bypass);
```

If you want, I can refactor these into modular function files (`rawToRGB.m`, `removeHotPixelsRAW.m`, etc.) to make them easier to unit-test and reuse.

---

## Attribution & Credits
- MATLAB Image Processing Toolbox
- (Optional) Deep Learning Toolbox for DnCNN denoiser
- Your code implements robust flat-fielding, PSF estimation and deconvolution building on standard image-processing techniques.

---

## Contributing
Pull requests welcome. Please:
1. Open an issue describing the feature/fix.
2. Add tests or example images if possible.
3. Document new parameters in README.

---

## Contact / Notes
If you want, I can:
- Convert the helper functions into separate `.m` files for cleaner structure, or
- Add a short quickstart demo script and example images you can commit to the repo, or
- Create a CHANGELOG.md and CONTRIBUTING.md.
