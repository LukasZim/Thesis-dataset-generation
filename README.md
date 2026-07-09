# Thesis Point Cloud Dataset Generation

Dataset generation pipeline for my MSc thesis **“Learning Unsigned Distance Fields to Simulate Brittle Fractures in Real-Time”** at TU Delft.

This repository contains the data generation part of the project. It creates impulse-dependent brittle fracture samples for a given mesh and converts the resulting fracture labels into surface-based unsigned distance fields (UDFs). These generated samples are used to train and evaluate machine learning models for predicting fracture patterns from impact information.

## Project context

The thesis investigates whether brittle fracture patterns can be learned on mesh surfaces in a way that is fast enough for real-time or interactive applications such as games. Instead of directly learning arbitrary fragment labels, the method learns an unsigned geodesic distance field over the mesh surface, where each vertex stores its distance to the nearest fracture boundary. The predicted distance field can then be segmented into fracture pieces.

## What this repository does

- Generates impulse-dependent fracture samples for individual meshes
- Samples impact positions and directions on mesh surfaces
- Uses fracture labels to identify fracture boundary vertices
- Computes unsigned geodesic distance fields on the mesh surface
- Stores generated data for downstream ML training and evaluation

## Dataset generation pipeline

The pipeline follows these main steps:

1. Load a watertight input mesh.
2. Generate fracture labels using an impulse-dependent fracture method.
3. Map fracture labels back to the original mesh when needed.
4. Detect vertices located on fracture boundaries.
5. Compute the unsigned geodesic distance to the nearest fracture boundary.
6. Save the resulting mesh, impulse, labels, and UDF values as training data.

## Thesis contribution

This code supports the dataset construction stage of the thesis. The generated data is used to train models that predict fracture-aware distance fields conditioned on impact location and direction.

## Related repository

The machine learning and evaluation part of the thesis is available here:

- `pointnet.pytorch` — ML models, training, evaluation, and segmentation experiments

## Keywords

`computer graphics` · `fracture simulation` · `mesh processing` · `point clouds` · `unsigned distance fields` · `dataset generation` · `real-time graphics`
