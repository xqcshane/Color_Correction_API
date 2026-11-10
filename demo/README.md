# 本地 API 测试网页

先启动颜色校正 API，确认 `http://localhost:5522/api/ready` 返回成功。在仓库根目录运行：

```bash
python -m demo.server
```

打开 `http://127.0.0.1:5533/`。这个网页服务器只使用 Python 标准库，不需要安装 GPU 或 API 后端依赖。它通过同源 `/api/*` 把上传和下载按块转发给固定的 API 地址，因此浏览器不需要另外配置 CORS。

前端 HTML/CSS/JS 位于 `demo/dist`，与生产 API 分开运行。生产 API 镜像不包含这些网页资产。页面的“使用示例文件”按钮在浏览器中生成 `synthetic-artwork-36dpi.pdf` 和 `synthetic-cmyk-lab-16-points.csv`，用于验证两条接口流程；这些配对是合成数据。

API 位于其他地址或要更换网页端口时：

```bash
python -m demo.server --api-base-url http://localhost:5522 --port 5533
```

也可以用环境变量 `API_BASE_URL` 设置默认 API 地址；命令行参数优先。默认只监听本机 `127.0.0.1`，需要改变监听地址时使用 `--host`。`GET /demo-config` 返回网页显示使用的 `apiBaseUrl`。

网页需要 PDF、同一测量组的 CMYK–Lab CSV 和准确的 PDF 图层名称。当前 API 没有检查图层列表的接口，图层名称需要手工输入。省略输出 DPI 就保留原稿密度；无法检测密度的纯矢量稿件需要手动指定 DPI。

预览由本地 PDF.js **6.5.438** 渲染到 canvas，主模块、worker 和许可证位于 `demo/dist/vendor`，不依赖 CDN。页面只绘制 PDF 第 1 页；超过 **80 MiB** 的原稿或结果需点击“加载 PDF 预览”后才读取预览。

结果预览通过支持 Range 的 URL 按需请求，并关闭后台自动获取剩余数据。只绘制第一页仍需解析 PDF 文档结构，实际读取量和浏览器内存取决于 PDF 内容及上游 Range 支持。完整结果 PDF 使用浏览器原生下载链接。

转发保留上传文件原始字节、PDF 下载状态和相关请求/响应头，支持大文件和 Range 下载。结果地址 `/api/jobs/{id}/result?preview=1` 会将响应改为 `inline`；普通结果地址仍按上游规则下载。服务只转发到启动时配置的 API，不接受由请求指定其他目标。

停止网页服务器时在终端按 `Ctrl+C`。API 服务单独运行、单独停止。
