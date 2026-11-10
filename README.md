# Color Correction API

从 CMYK Recipe Studio 提取的独立 PDF 颜色校正服务。提供两个 GPU 接口，输入 PDF、CMYK–Lab CSV、需要处理的图层名称和可选输出 DPI，输出 PDF。默认使用精细算法和原稿密度，不生成布料/设备右下角标注。仓库另附独立运行的本地测试网页。

## 部署（Linux + RTX 3090）

需要 Docker Engine、Docker Compose v2、可用 NVIDIA 驱动和 NVIDIA Container Toolkit。基础镜像是 `nvidia/cuda:12.6.3-devel-ubuntu24.04`，包含 CuPy 所需的 NVRTC 和 CUDA 头文件；宿主机无需另装 CUDA Toolkit。建议 Linux NVIDIA 驱动不低于 **560.35.05**。

RTX 3090 使用 Ampere **compute capability 8.6 / sm_86**。服务用 CuPy 检测实际显卡并编译内核，不包含 RTX 4090 专属设置。默认 GPU 设备编号为 `0`；多卡服务器可在 `.env` 中改 `GPU_DEVICE_ID`。

```bash
git clone https://github.com/xqcshane/Color_Correction_API.git
cd Color_Correction_API
cp .env.example .env
bash deploy.sh
```

成功后接口地址为 `http://服务器IP:5522`，交互文档位于 `/docs`，OpenAPI 位于 `/openapi.json`。

```bash
# 更新当前跟踪分支、构建并部署（本地有改动则停止，不覆盖改动）
bash deploy.sh --pull

# 使用现有镜像重新创建容器，同时应用 .env，保留上传与结果
bash restart.sh

# 首次编译较慢时，延长等待服务就绪的秒数
HEALTH_TIMEOUT=600 bash deploy.sh

# 状态与日志
docker compose ps
docker compose logs -f --tail=100 color-api
curl --fail http://127.0.0.1:5522/api/ready
curl --fail http://127.0.0.1:5522/api/capabilities
```

`deploy.sh` 构建镜像后等待 Docker 健康检查成功，`restart.sh` 不重新构建镜像。健康检查访问 `/api/ready`，GPU 检查会执行并同步 CUDA 内核。GPU 不可用时，生产接口返回 503，不会自动以 CPU 完成生产任务。

## 本地测试网页

先确保 `http://localhost:5522/api/ready` 返回成功，再在仓库根目录运行：

```bash
python -m demo.server
```

打开 `http://127.0.0.1:5533/`。网页服务器只需 Python 标准库，无需另装 GPU 依赖；GPU 计算由已启动的 API 完成。前端 HTML/CSS/JS 位于 `demo/dist`，通过独立的同源代理流式上传和下载。生产 API 镜像不包含 demo。更换 API 地址、端口等用法见 [demo 说明](demo/README.md)。

“使用示例文件”生成合成 PDF 和合成 CMYK–Lab CSV，仅供测试流程。预览使用本地 PDF.js **6.5.438**，不依赖 CDN，只绘制第一页；超过 **80 MiB** 的文件需要手动加载预览。结果预览通过 Range 按需请求，实际读取量和浏览器内存取决于 PDF 结构。完整结果 PDF 使用浏览器原生下载。

## 两个 PDF 接口

| 接口 | 作用 |
| --- | --- |
| `POST /api/nearest` | GPU 找最接近：按 CIEDE2000 搜索实测色块，采用匹配色块的 CMYK |
| `POST /api/predict` | GPU 推算新色：根据实测 CMYK–Lab 配对拟合前向模型，再反求新的 CMYK |

两个接口都使用 `multipart/form-data`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `pdf` | 文件，必填 | 带可选内容图层（OCG）的 PDF |
| `csv` | 文件，必填 | 同一组印花条件的 CMYK–Lab 配对 CSV |
| `layers` | 字符串，必填 | 需要处理的图层名的 JSON 数组，例如 `["印花图层","纹理"]`；精确匹配名称 |
| `dpi` | 数值，可选 | 输出 DPI；省略时保留原稿密度。若使用 px/mm，乘 `25.4` 得到 DPI，例如 `12 px/mm = 304.8 DPI` |

成功响应为 `application/pdf`。没有 `quality` 或标注参数；处理精度固定为精细。其他图层保留，PDF 页数、方向和物理尺寸保持。

需要 PDF 中存在真实 OCG 图层。没有图层、层名不存在或同一个名称对应多个图层时，接口会拒绝请求，避免误处理其他内容。

### GPU 找最接近

```bash
curl --fail-with-body --max-time 0 \
  -X POST http://127.0.0.1:5522/api/nearest \
  -F 'pdf=@input.pdf;type=application/pdf' \
  -F 'csv=@cmyk-lab.csv;type=text/csv' \
  -F 'layers=["印花图层"]' \
  -o nearest.pdf
```

### GPU 推算新色（显式 304.8 DPI）

```bash
curl --fail-with-body --max-time 0 \
  -X POST http://127.0.0.1:5522/api/predict \
  -F 'pdf=@input.pdf;type=application/pdf' \
  -F 'csv=@cmyk-lab.csv;type=text/csv' \
  -F 'layers=["印花图层"]' \
  -F 'dpi=304.8' \
  -o predicted.pdf
```

省略 `dpi` 时，能直接替换的嵌入图像保留原始像素。显式指定 `dpi` 时，安全的图像图层根据每张图像实际放置尺寸及旋转后的图像轴长度计算目标像素，重采样后仍写回原图像对象，保留位置、图层和层叠顺序；透明蒙版同步重采样并写入独立副本，不影响其他对象共享的原蒙版。纯矢量或其他无法检测原始密度的稿件需要明确指定 `dpi`，不会静默改成预览分辨率。

无法直接替换的矢量/混合内容或超过原生像素阈值的图像，按输出密度分块重绘。重绘时仅渲染指定层，未选层及无图层内容的原始对象保持；对于各层使用独立内容流的稿件，新增处理图像插入原选层位置，保留背景和辅助线的前后关系。若需要重绘的同一内容流或 Form XObject 混合多个层，或选层与其他内容交错绘制，整层重绘无法保证层叠顺序，接口会明确拒绝。此时应重新导出各图层独立的内容流；可直接替换的图像采用原生对象路径时不受这项层叠限制。

同一个嵌入图像对象若在多个不同物理尺寸下复用，无法同时让所有位置满足统一的显式 DPI。接口会要求省略 `dpi` 保留原始像素，或将各放置位置导出为独立图像对象。

## CMYK–Lab CSV

支持 UTF-8（含 BOM）和 GB18030。标准表头是：

```csv
C,M,Y,K,L,a,b
```

也支持原项目导出的中文表头，如 `布上标签C (%)`、`平均仪器L*` 等。CMYK 采用 0–100 百分数，L 为 0–100，a/b 为有限数值。`nearest` 至少需要 1 个不同 CMYK 配对，`predict` 至少需要 8 个不同 CMYK 配对。同一 CMYK 的重复测量会按 `count` / 读数数量 / 重复数量加权求平均。

每次只能提交同一布料、设备和测量条件的数据。CSV 可省略布料等元数据；若提供了布料、设备、批次、测量组、光源等元数据且包含多个分组，接口会拒绝，需要先拆成独立 CSV。存在无效数据行时返回 422，不会静默跳过。CSV 中的布料和设备字段仅用于数据分组校验，不会写到 PDF。

## 大文件与异步任务

直接调用会持续等待并下载 PDF。大稿件耗时较长时，使用同一个接口加 `?wait=false`，先返回 202 和任务 ID，避免中间代理的 HTTP 等待超时：

```bash
curl --fail-with-body \
  -X POST 'http://127.0.0.1:5522/api/nearest?wait=false' \
  -F 'pdf=@input.pdf;type=application/pdf' \
  -F 'csv=@cmyk-lab.csv;type=text/csv' \
  -F 'layers=["印花图层"]'

# 用返回的 id 替换 JOB_ID；status=succeeded 后下载
curl --fail http://127.0.0.1:5522/api/jobs/JOB_ID
curl --fail http://127.0.0.1:5522/api/jobs/JOB_ID/result -o corrected.pdf

# 取消任务 / 删除已结束任务
curl --fail -X POST http://127.0.0.1:5522/api/jobs/JOB_ID/cancel
curl --fail -X DELETE http://127.0.0.1:5522/api/jobs/JOB_ID
```

默认只使用一个 GPU 工作线程，其他任务排队，防止单张 3090 被并发模型占满显存。容器重建保留已完成结果；重启时尚未完成的任务会标记失败，需要重新提交。

上传、结果和临时文件写到 Docker 命名卷 `runtime-data` 对应的 `/data`，不把整份上传 PDF 留在 Python 内存里。PDF 按块处理；单张原生嵌入图像或其重采样目标超过 6400 万像素时改走分块渲染，输出图像流逐块压缩。PDF 解码与解析仍会占用主机内存，实际可处理规模取决于稿件结构、RAM 和磁盘空间，不能用显存容量代替这些资源。

默认上传 PDF 上限 64 GiB、CSV 上限 64 MiB，可在 `.env` 调整。PDF 路径使用 PyMuPDF 解码图像，不经过 Pillow，因此不受 Pillow 默认大图像素限制影响。pypdf 仅解析 PyMuPDF 已解压的绘制指令和字典，按需查询 PDF 对象，不用 `PdfReader` 将整份文件读入内存，也不走 pypdf 的图像解码流程。绘制指令的解析预算由默认 200 万提高到 **2000 万 token**，数组及字典限制提高到 **100 万项**；token 预算还可通过 `.env` 增大。上述预算约束指令复杂度，与 PDF 嵌入图像像素量不同。

若前面另加反向代理，也需按实际文件大小配置上传限制和超时；异步模式适合长时间处理。

## 配置

修改 `.env` 后执行 `bash restart.sh`。

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `BIND_HOST` | `0.0.0.0` | 宿主机监听地址；端口固定 5522 |
| `GPU_DEVICE_ID` | `0` | 宿主机 GPU 编号或 UUID |
| `CMYK_API_MAX_UPLOAD_GIB` | `64` | 单个 PDF 上传大小上限 |
| `CMYK_API_MAX_CSV_MIB` | `64` | 单个 CSV 上传大小上限 |
| `CMYK_API_UPLOAD_CHUNK_MIB` | `8` | 上传拷贝块大小 |
| `CMYK_API_MAX_PENDING_JOBS` | `4` | 任务容量限制 |
| `CMYK_API_JOB_TTL_HOURS` | `24` | 已结束任务保留时间；到期清理后无法下载 |
| `CMYK_PDF_NATIVE_MAX_PIXELS` | `64000000` | 单张原始图像或重采样目标的原生处理阈值，超过时使用分块渲染 |
| `CMYK_PDF_PARSE_MAX_TOKENS` | `20000000` | 单个绘制内容流的 pypdf 解析 token 预算 |

`CMYK_API_WORK_DIR`、`CUPY_CACHE_DIR`、`TMPDIR` 已在容器中指向 `/data`。Compose 生产配置固定 `CMYK_API_FORCE_CPU=false`。

## 开发与测试

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r api/requirements-dev.txt
CMYK_API_FORCE_CPU=true python -m pytest tests -q

# 仅用于没有 GPU 的本地功能验证
CMYK_API_FORCE_CPU=true python -m api.run

# 裸机 GPU 开发另装 CuPy，宿主机需有 CUDA 12.x Toolkit
python -m pip install -r api/requirements-gpu.txt
python -m api.run
```

服务采用 NumPy/CuPy 进行颜色计算；PDF 解析、分层合成和写文件由 CPU 完成。新生成的处理图像使用 DeviceCMYK，不额外嵌入 ICC；未选对象的原始数据保持。推算新色是基于测量数据的预测结果；生产颜色效果需用实际打印测量确认。

## 依赖与部署依据

- [NVIDIA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus)：RTX 3090 为 8.6。
- [CUDA 12.6.3 发行说明](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-toolkit-release-notes/index.html)：对应驱动版本。
- [CuPy 13.6 安装说明](https://docs.cupy.dev/en/v13.6.0/install.html)：支持 CUDA 12.6、Python 3.12，CUDA 12.2 以后需要编译头文件。
- [Docker Compose GPU 配置](https://docs.docker.com/compose/how-tos/gpu-support/)：通过 `device_ids` 和 `capabilities: [gpu]` 分配显卡。
