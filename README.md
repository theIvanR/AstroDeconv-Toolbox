# AstroDeconv Toolbox
**Planet Deconvolver** & **Deep Sky Deconvolver** — preprocessing + deconvolution tools for astronomical imaging

Two MATLAB preprocessing/deconvolution scripts intended to maximize SNR and extract as much detail as possible from astronomical images:

- `deepsky_deconvolving_preprocessor.m` — tuned for faint, extended targets (nebulae, galaxies).
- `planetary_deconvolving_preprocessor.m` — tuned for bright, high-SNR small-disk targets (planets, Moon, bright satellites).

> Why two tools? SNR and the dominant error modes differ between planetary and deep-sky imaging. Planet code assumes high signal, compact PSF, and benefits most from per-channel Wiener deconvolution. Deep-sky code focuses on low SNR, Poisson noise, careful regularization and PSF confidence weighting.

---

# Overview
These scripts form a small toolbox that:

- Converts RAW (or pre-stacked) images to linear RGB when needed.
- Builds robust master flats and confidence-weighted inverse flats (deep-sky pipeline).
- Applies robust per-channel or luminance-based deconvolution (planetary pipeline).
- Contains hooks for hot-pixel removal, auto white balance, gaussian low-pass filtering, CNN denoising (optional), PSF estimation from nearby star patches, and (semi) blind deconvolution.
- Saves high-dynamic-range results as FP32 TIFFs (or clipped uint16 if desired).

The goal is to **maximize final SNR by improving data quality** (better flat correction, PSF estimation, denoising, and tailored deconvolution) — because SNR scales linearly with raw signal quality, but only with the square root of the number of frames.

---

# Features
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

# Requirements
- MATLAB R2020a or later (tested with R2022a+ features)
- Image Processing Toolbox
- (Optional) Deep Learning Toolbox — for `denoisingNetwork("DnCNN")`
- `rawread` or other RAW reader if you plan to convert `.ARW` inside the pipeline
- (Optional) Parallel Computing Toolbox for `parfor` speedups

---

# Repository layout (suggested)
