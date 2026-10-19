# Verification of the prepared public package

Verified locally on 2026-10-19 using the supplied Visual Studio solution.

- Both **Release | x64** and **Debug | x64** built successfully.
- MSVC compiler: 19.44.35222; toolset: v143 / 14.44.35207.
- OpenCV: 4.12.0, MSVC x64 libraries.
- Source language standard: C++20.
- Release executable completed all 600 test images, including all 200 ring-adhesion images; no image evaluation was skipped.
- No timing result is asserted by this verification.

| Category | Evaluated images | Successful/GT particles | Success rate | Mean IoU | Penalized Feret relative error |
|---|---:|---:|---:|---:|---:|
| two_particle | 200 | 392/400 | 98.00% | 96.86% | 2.45% |
| three_particle | 200 | 577/600 | 96.17% | 96.40% | 3.11% |
| ring_particle | 200 | 764/791 | 96.59% | 96.16% | 2.83% |

Overall particle success rate: **1733/1791 = 96.76%**. The IoU and Feret columns above retain the evaluation definitions implemented in the source.

The only change to the supplied C++ source during package preparation was replacing `ring_shaped` with `ring_particle` in the dataset category list and its missing-dataset error message. No segmentation, matching, distance-scoring, or timing logic was changed.

Source SHA-256 (`src/Skeleton-Constrained Method.cpp`):

```text
0658e5547233febfc247c461dceef860528af33679716c946647b6bf9d52e61e
```

This SHA-256 identifies this file's bytes; it is not a Git commit. Record the actual full Git commit after committing the final repository contents.
