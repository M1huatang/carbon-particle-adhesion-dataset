# 粘连碳颗粒分割：源码与数据

## 目录

- `src`：本文算法源码和 `json.hpp`。
- `data`：两颗、三颗、环形三类测试数据，每类200张，均含对应JSON标注。
- `yolo11m-seg`：独立的YOLO训练、验证数据，与 `src`、`data` 并列；训练540张，验证136张，只保留一份。
- `SkeletonConstrained.sln` 和 `SkeletonConstrained.vcxproj`：由提供的aodians工程整理的Visual Studio工程，编译目标为 `src/Skeleton-Constrained Method.cpp`。

## 如何编译和运行

1. 安装Visual Studio 2022及“使用C++的桌面开发”工作负载、v143工具集和Windows SDK。
2. 安装OpenCV 4.12.0的Windows x64 MSVC版本。
3. 将 `OpenCV.local.props.example` 复制为 `OpenCV.local.props`，将其中的 `OpenCVRoot` 改为本机OpenCV目录。该目录下应直接存在 `include` 和 `vc16`。例如官方压缩包解压后的 `C:\opencv\build`。不要将本机路径配置文件上传GitHub。
4. 双击 `SkeletonConstrained.sln`，选择 `Release | x64`，生成解决方案。也可选择 `Debug | x64`；工程已分别链接相应版本的OpenCV库。
5. 在仓库根目录的PowerShell中执行 `.\build\x64\Release\SkeletonConstrained.exe`。程序会处理三类测试图像，结果保存在 `results`。

工程会将相应OpenCV DLL自动复制到可执行文件旁边。无需上传整个OpenCV、Visual Studio、`.vs`、编译缓存或生成的结果；`.gitignore` 已排除这些文件。

## 版本和核验

已核实环境：Visual Studio Community 2022 17.14.22（17.14.36804.6），MSVC 19.44.35222，v143工具集（14.44.35207），C++20，OpenCV 4.12.0，nlohmann/json 3.12.0。

本发布工程Release和Debug均已编译通过；Release已完整运行600张测试图像。各类别结果及源码SHA-256见 `VERIFICATION.md`。此次源码仅修改两处 `ring_shaped` 为 `ring_particle`，算法和评价逻辑未作修改。

YOLO文件夹保留的是LabelMe JSON标注，尚非Ultralytics直接读取的TXT标签。本工程只运行本文C++算法；若需复现YOLO训练，还需提供相应标签转换脚本、数据YAML、训练配置和命令。

最终上传并提交后，用 `git rev-parse HEAD` 取得完整40位commit，填写到论文或审稿回复中。文件SHA-256不能代替commit。
