# Adhered Carbon Particle Segmentation

This repository contains the C++ source code and test data for the skeleton-topology-constrained segmentation method, together with the YOLO11m-seg training/validation data, comparison checkpoint and corresponding training configuration.

## Repository layout

```text
src/
  Skeleton-Constrained Method.cpp
  json.hpp
data/
  two_particle/{images,annotations}/
  three_particle/{images,annotations}/
  ring_particle/{images,annotations}/
yolo11m-seg/
  images/{train,val}/
  jsons/{train,val}/
  train/
    weights/best.pt
    args.yaml
SkeletonConstrained.sln
SkeletonConstrained.vcxproj
OpenCV.local.props.example
```

## Datasets

| Dataset | Images | Image directory | Annotation directory |
|---|---:|---|---|
| Two-particle adhesion test set | 200 | [`data/two_particle/images/`](data/two_particle/images/) | [`data/two_particle/annotations/`](data/two_particle/annotations/) |
| Three-particle adhesion test set | 200 | [`data/three_particle/images/`](data/three_particle/images/) | [`data/three_particle/annotations/`](data/three_particle/annotations/) |
| Ring-shaped adhesion test set | 200 | [`data/ring_particle/images/`](data/ring_particle/images/) | [`data/ring_particle/annotations/`](data/ring_particle/annotations/) |
| YOLO training set | 540 | [`yolo11m-seg/images/train/`](yolo11m-seg/images/train/) | [`yolo11m-seg/jsons/train/`](yolo11m-seg/jsons/train/) |
| YOLO validation set | 136 | [`yolo11m-seg/images/val/`](yolo11m-seg/images/val/) | [`yolo11m-seg/jsons/val/`](yolo11m-seg/jsons/val/) |

The test set contains 600 images in total. Annotations are stored as LabelMe JSON polygons. For YOLO retraining, convert the JSON annotations to Ultralytics segmentation TXT labels and prepare a dataset YAML with the appropriate local paths.

## Build and run the C++ method

Requirements: Windows x64, Visual Studio 2022 with the Desktop development with C++ workload and v143 toolset, C++20, and OpenCV 4.12.0 (Windows x64 MSVC package). The required `json.hpp` header is included in `src/`.

1. Copy `OpenCV.local.props.example` to `OpenCV.local.props` and set `OpenCVRoot` to the local OpenCV build directory, such as `C:\opencv\build`. This directory should contain `include` and `vc16`.
2. Open `SkeletonConstrained.sln` in Visual Studio, select **Release | x64**, and build the solution.
3. Run the executable from the repository root:

   ```powershell
   .\build\x64\Release\SkeletonConstrained.exe
   ```

The program reads the images and annotations in the three `data` categories. Segmentation images, predicted JSON annotations and evaluation CSV files are written to `results/<category>/`. The project copies the required OpenCV DLL beside the executable during the build.

## YOLO11m-seg checkpoint

- Comparison checkpoint: [`yolo11m-seg/train/weights/best.pt`](yolo11m-seg/train/weights/best.pt).
- Corresponding training configuration: [`yolo11m-seg/train/args.yaml`](yolo11m-seg/train/args.yaml).

Both files originate from the same training run. The `data` and `save_dir` entries in `args.yaml` retain the original training-workspace paths; adapt these paths when using the configuration locally. The C++ project implements the proposed segmentation method.

## Fixed archive version

The source code, datasets, YOLO checkpoint and training configuration are available at the [fixed repository revision `45d96e9e368422249e7e9a12ba170d6e0990f46d`](https://github.com/M1huatang/carbon-particle-adhesion-dataset/tree/45d96e9e368422249e7e9a12ba170d6e0990f46d).
