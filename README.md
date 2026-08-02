# AstroDeconv Toolbox

**Astronomical image preprocessing and deconvolution toolbox for deep-sky and planetary imaging.**

AstroDeconv prepares individual astronomical exposures before stacking by correcting sensor artifacts, calibration errors, noise, and optical degradation.

The goal is to provide a modular framework for improving the quality of individual light frames before integration into a final stacked image.

---

# Overview

Astronomical images are affected by multiple sources of degradation before stacking:

* sensor defects
* illumination non-uniformity
* photon and electronic noise
* atmospheric turbulence
* optical aberrations
* tracking and alignment errors

AstroDeconv provides preprocessing tools to identify, model, and compensate for these effects.

---

# Problems Addressed

## Sensor Artifacts

### Hot pixels and defective pixels

Removes isolated sensor defects and anomalous high-intensity pixels using local statistical methods.

### Impulse noise

Suppresses isolated pixel outliers caused by sensor readout errors and transient artifacts.

### Color channel artifacts

Detects abnormal RGB channel deviations and suppresses false color artifacts while preserving astronomical structures.

---

## Calibration Artifacts

### Flat-field correction

Corrects pixel sensitivity variations and illumination gradients using master flat-field frames.

Features:

* robust flat-frame stacking
* kappa-sigma rejection
* monochrome and RGB support

### Optical non-uniformity

Models illumination effects including:

* vignetting
* dust shadows
* sensor/optical response variations

---

## Optical Degradation

### Atmospheric seeing

Models atmospheric blur through point spread function (PSF) estimation and simulation.

### PSF estimation

The toolbox extracts stellar PSF information directly from astronomical images.

Current measurements include:

* stellar centroid alignment
* FWHM estimation
* ellipticity
* radial PSF variation
* coma-like asymmetry estimation

### Deconvolution

Uses estimated PSF kernels for image restoration before stacking.

---

## Noise Reduction

AstroDeconv addresses:

* photon shot noise
* sensor read noise
* low signal-to-noise structures

A neural denoising stage is included as part of the preprocessing pipeline.

---

# Processing Pipeline

Typical workflow:

```
RAW astronomical exposures
          |
          v
     RAW conversion
          |
          v
    Individual TIFF frames
          |
          v
    Artifact correction
          |
          v
    Flat-field calibration
          |
          v
    Noise reduction
          |
          v
    PSF estimation
          |
          v
    Deconvolution
          |
          v
  Preprocessed frames
          |
          v
       Stacking
```

AstroDeconv focuses on improving individual frames before stacking rather than replacing the stacking process itself.

---

# Synthetic Dataset Generator

The repository contains a synthetic astronomical image generator for controlled testing.

The simulator can generate:

* star fields
* nebulosity
* flat-field response
* vignetting
* dust attenuation
* atmospheric PSF blur
* telescope drift
* shot noise
* read noise
* hot pixels
* saturation effects

Synthetic data allows preprocessing algorithms to be evaluated against known ground truth.

Generated dataset structure:

```
output/
|
├── Light/
|     simulated astronomical exposures
|
├── Flat/
|     simulated flat-field frames
|
└── truth/
      ground-truth scene and instrument response
```

---

# Repository Structure

```
AstroDeconv/
|
├── main/
|   Main preprocessing pipeline
|
|   ├── main.m
|   └── functions/
|       Calibration, artifact removal,
|       PSF estimation and restoration
|
├── auxiliary/
|
|   ├── convert_raws/
|   |     RAW conversion utilities
|   |
|   ├── fix_if_broken_16bit.py
|   |     TIFF compatibility tools
|   |
|   └── gen_synth_data/
|         Synthetic dataset generator
|
└── README.md
```

---

# Requirements

## MATLAB

Required:

* MATLAB
* Image Processing Toolbox
* Deep Learning Toolbox
* Parallel Computing Toolbox

## External tools

RAW conversion requires:

```
dcraw_emu
```

---

# Quick Start

## 1. Generate synthetic data

From MATLAB:

```matlab
cd auxiliary/gen_synth_data
synthetic_main
```

This generates test light and flat frames.

---

## 2. Prepare input data

Place converted frames into:

```
data/

├── Light/
|     individual exposures
|
└── Flat/
      flat-field exposures
```

---

## 3. Run preprocessing

From MATLAB:

```matlab
cd main
main
```

The resulting preprocessed images can then be passed to an external stacking workflow.

---

# Project Status

AstroDeconv is an experimental research toolbox.

Implemented:

* flat-field calibration
* sensor artifact removal
* RGB artifact correction
* neural denoising preprocessing
* stellar PSF extraction
* Richardson-Lucy deconvolution
* synthetic astronomical image generation

Future work:

* spatially varying PSF models
* GPU acceleration
* improved optical aberration modeling
* automated stacking integration

---

# License

License information will be added here.
