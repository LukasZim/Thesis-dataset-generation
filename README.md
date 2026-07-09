# Thesis Point Cloud Dataset Generation

Dataset-generation pipeline for my MSc thesis **“Learning Unsigned Distance Fields to Simulate Brittle Fractures in Real-Time”** at TU Delft.

This repository contains the data-generation part of the project. It creates impulse-dependent brittle fracture samples for a given mesh and converts the resulting fracture labels into surface-based unsigned distance fields (UDFs). These samples are then used to train machine learning models for fracture prediction.

## Pipeline overview

The dataset pipeline generates fractures for a mesh under different impact configurations, maps the resulting labels back to the original mesh, detects fracture-boundary vertices, and computes an unsigned geodesic distance field over the mesh surface.

![Dataset generation pipeline](docs/images/pipeline_overview.png)

## Unsigned distance field generation

Instead of directly learning arbitrary fracture labels, the thesis represents fracture patterns as a distance field. Vertices close to fracture boundaries receive low UDF values, while vertices farther from cracks receive higher values.

![Unsigned distance field generation](docs/images/udf_generation.png)

## Related repository

The machine learning and evaluation code is available in the companion repository:

- [`pointnet.pytorch`](https://github.com/LukasZim/pointnet.pytorch)
