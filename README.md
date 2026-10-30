# Adhered carbon particle segmentation

Source code, test images and annotations for the skeleton-constrained segmentation method, together with the separate YOLO11m-seg training/validation data used for the comparison experiment.

## Repository layout

```text
SkeletonConstrained.sln
SkeletonConstrained.vcxproj
OpenCV.local.props.example
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
```

The test dataset contains 600 images: 200 in each adhesion category. Images have matching LabelMe JSON polygon annotations. The separate YOLO dataset contains 540 training and 136 validation images, each with matching LabelMe JSON annotations. Only one copy of this training/validation dataset is included. These JSON annotations are not YOLO segmentation TXT labels: conversion to the Ultralytics polygon-label format and a dataset YAML are needed before YOLO training. This C++ solution implements the proposed method; it does not train YOLO.

## Build environment

- Windows x64; Visual Studio Community 2022, version 17.14.22 (17.14.36804.6).
- MSVC C/C++ compiler 19.44.35222, platform toolset v143 (installed tools directory 14.44.35207).
- C++20; Release uses `/O2` and `/MD`; Debug uses `/Od` and `/MDd`.
- OpenCV 4.12.0, Windows x64 MSVC package (`opencv_world4120.lib` / `opencv_world4120.dll` for Release; `opencv_world4120d.lib` / `opencv_world4120d.dll` for Debug).
- nlohmann/json 3.12.0, included as `src/json.hpp`; retain its embedded MIT license notice.

## Build with Visual Studio

1. Install Visual Studio 2022 with the **Desktop development with C++** workload, the v143 toolset, and a Windows 10/11 SDK.
2. Install the OpenCV 4.12.0 Windows x64 MSVC package. Locate the directory containing both `include/opencv2/opencv.hpp` and `vc16/lib/opencv_world4120.lib` (normally the extracted `build` directory).
3. Copy `OpenCV.local.props.example` to `OpenCV.local.props`, and set `OpenCVRoot` to that directory. This local configuration is ignored by Git. Alternatively, place the directory contents under `OpenCV/` beside the solution, or set an `OpenCVRoot` environment variable before starting Visual Studio.
4. Open `SkeletonConstrained.sln`, select **Release | x64**, and choose **Build Solution**. For Debug, select **Debug | x64**; the project links only the matching Debug library.
5. The executable and required OpenCV world DLL are placed in `build/x64/Release/` (or `build/x64/Debug/`). The project copies the DLL automatically.

From a Visual Studio Developer PowerShell, the equivalent Release build is:

```powershell
msbuild .\SkeletonConstrained.sln /m /p:Configuration=Release /p:Platform=x64
```

## Run

Keep the repository layout intact and run from its root:

```powershell
.\build\x64\Release\SkeletonConstrained.exe
```

The program reads the three `data` categories, processing numeric image IDs 1 through 200 in JPG format and their JSON annotations. It discovers the repository root from the working directory or executable location. It writes segmentation images, predicted JSON annotations and CSV summaries under `results/<category>/`. Runtime CSV output is part of the supplied source's existing behavior; preparation of this package did not change its timing or scoring code.

## Exact source revision

After committing and uploading the final verified files, obtain the full 40-character revision with:

```bash
git rev-parse HEAD
```

Cite that revision and its GitHub `/tree/<full-commit-SHA>` URL in the paper/reviewer response. A source-file SHA-256 is a file-integrity check, not a Git commit. A public-release commit created now must not be described as a historical experimental commit if no such record exists.
