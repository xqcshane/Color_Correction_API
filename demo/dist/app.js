"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const el = Object.fromEntries([
    "jobForm", "pdfFile", "csvFile", "pdfDrop", "csvDrop", "pdfName", "pdfMeta", "csvName", "csvMeta",
    "loadSample", "sampleNotice", "layers", "densityMode", "dpi", "densityHint", "formError", "submitJob", "submitText",
    "originalTab", "resultTab", "previewStage", "previewEmpty", "previewEmptyTitle", "previewEmptyDetail", "loadPreview", "pdfPreview", "pdfCanvas", "previewPageInfo", "previewLoading",
    "previewFilename", "previewKind", "taskHeading", "progressValue", "progressBar", "progressFill", "taskMessage",
    "stepUpload", "stepQueue", "stepProcess", "stepDone", "resultMetrics", "resultDevice", "resultDensity", "resultTime",
    "jobId", "retryPoll", "cancelJob", "newJob", "downloadPdf", "connectionBadge", "connectionText", "serviceDot",
    "serviceDetails", "refreshConnection", "apiAddress", "docsLink", "requestPreview", "endpointLabel",
  ].map((id) => [id, byId(id)]));
  const STORAGE_PREFIX = "color-correction-demo.current-job.v2:";
  const MANUAL_PREVIEW_BYTES = 80 * 1024 * 1024;
  const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);
  const state = {
    gpuReady: false, capabilities: null, apiBaseUrl: "http://localhost:5522", storageKey: null, connecting: false,
    pdf: null, csv: null, originalUrl: null, resultUrl: null, resultBytes: 0,
    view: "original", originalApproved: false, resultApproved: false,
    busy: false, phase: "idle", jobId: null, record: null, options: null,
    xhr: null, pollTimer: null, pollSequence: 0, generation: 0, startedAt: null, cancelRequested: false,
  };
  const methodInput = () => document.querySelector('input[name="method"]:checked');
  const text = (node, value) => { node.textContent = String(value ?? ""); };
  const sizeText = (bytes) => bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(2)} GiB`
    : bytes >= 1024 ** 2 ? `${(bytes / 1024 ** 2).toFixed(1)} MiB`
      : `${Math.max(1, Math.ceil(bytes / 1024))} KiB`;
  const round = (number) => Number(Number(number).toFixed(1));
  const preview = { generation: 0, url: null, loadingTask: null, renderTask: null, document: null, page: null, status: "idle" };
  let pdfModulePromise = null;
  let resizeTimer = null;

  function showError(message) {
    text(el.formError, message);
    el.formError.hidden = !message;
  }

  function friendlyError(payload, fallback = "请求失败，请稍后重试。") {
    const detail = payload?.detail ?? payload;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return detail.map((item) => item?.msg || "请求字段不正确").join("；");
    if (detail && typeof detail === "object") return detail.error || detail.message || fallback;
    return fallback;
  }

  async function fetchJson(url, options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal, cache: "no-store" });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        const error = new Error(friendlyError(payload, `服务返回 HTTP ${response.status}`));
        error.status = response.status;
        throw error;
      }
      if (!payload || typeof payload !== "object") throw new Error("服务返回了无法识别的响应。");
      return payload;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("连接超时，请检查服务后重试。");
      if (error instanceof TypeError) throw new Error("无法连接 API 服务，请检查网络或服务地址。");
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }

  function readLayers() {
    const raw = el.layers.value.trim();
    let names;
    if (raw.startsWith("[")) {
      try { names = JSON.parse(raw); } catch { throw new Error("图层 JSON 格式无效，请使用名称数组或每行一个名称。"); }
    } else {
      names = raw.split(/\r?\n/).map((name) => name.trim()).filter(Boolean);
    }
    if (!Array.isArray(names) || !names.length || names.some((name) => typeof name !== "string" || !name.trim())) {
      throw new Error("请填写至少一个需要处理的图层名称。");
    }
    if (new Set(names).size !== names.length) throw new Error("图层名称有重复，请每个名称只填写一次。");
    if (new TextEncoder().encode(JSON.stringify(names)).length > 64000) throw new Error("图层名称内容过长。");
    return names;
  }

  function readOptions() {
    const options = { method: methodInput().value, layers: readLayers(), densityMode: el.densityMode.value };
    if (options.densityMode === "custom") {
      options.dpi = Number(el.dpi.value);
      if (!el.dpi.value.trim() || !Number.isFinite(options.dpi) || options.dpi <= 0) {
        throw new Error("自定义 DPI 需要填写大于 0 的数字。");
      }
    }
    return options;
  }

  function saveCurrentJob() {
    if (!state.storageKey) return;
    try {
      const options = state.options || readOptions();
      localStorage.setItem(state.storageKey, JSON.stringify({ jobId: state.jobId, options }));
    } catch { /* Storage restrictions must not interrupt a processing job. */ }
  }

  function updateRequestPreview() {
    const method = methodInput().value;
    let layers;
    try { layers = readLayers(); } catch { layers = el.layers.value; }
    const fields = { pdf: state.pdf?.name || "<PDF 文件>", csv: state.csv?.name || "<CMYK–Lab CSV>", layers: JSON.stringify(layers) };
    if (el.densityMode.value === "custom") fields.dpi = el.dpi.value;
    text(el.endpointLabel, `POST /api/${method}`);
    text(el.requestPreview, JSON.stringify({ endpoint: `/api/${method}?wait=false`, contentType: "multipart/form-data", fields }, null, 2));
  }

  function updateControls() {
    el.jobForm.querySelectorAll("input, textarea, select").forEach((input) => { input.disabled = state.busy; });
    el.dpi.disabled = state.busy || el.densityMode.value !== "custom";
    el.loadSample.disabled = state.busy;
    el.submitJob.disabled = state.busy || !state.gpuReady;
    text(el.submitText, state.busy ? (state.phase === "uploading" ? "正在上传文件…" : "GPU 任务处理中…") : "开始 GPU 处理");
    el.cancelJob.hidden = !state.busy;
    el.cancelJob.disabled = state.cancelRequested;
    text(el.cancelJob, state.phase === "uploading" ? "取消上传" : state.cancelRequested ? "正在取消…" : "取消任务");
    el.newJob.hidden = state.busy || state.phase === "idle";
    el.originalTab.disabled = !state.pdf;
    el.resultTab.disabled = !state.resultUrl;
    text(el.densityHint, el.densityMode.value === "custom"
      ? "只重采样指定图层，页面物理尺寸保持不变。" : "保持原稿像素密度与页面物理尺寸。");
    updateRequestPreview();
  }

  function setProgress(percent, step, error = false) {
    const value = Math.max(0, Math.min(100, Math.round(percent)));
    el.progressValue.replaceChildren(document.createTextNode(String(value)), Object.assign(document.createElement("span"), { textContent: "%" }));
    el.progressFill.style.width = `${value}%`;
    el.progressBar.setAttribute("aria-valuenow", String(value));
    el.progressBar.classList.toggle("has-error", error);
    [el.stepUpload, el.stepQueue, el.stepProcess, el.stepDone].forEach((node, index) => {
      node.classList.toggle("active", index === step && !error);
      node.classList.toggle("done", index < step || (step === 3 && value === 100));
    });
  }

  function destroyPreview() {
    preview.generation += 1;
    preview.renderTask?.cancel();
    const task = preview.loadingTask;
    preview.loadingTask = null;
    preview.renderTask = null;
    preview.document = null;
    preview.page = null;
    preview.url = null;
    preview.status = "idle";
    if (task) Promise.resolve(task.destroy()).catch(() => {});
    el.previewLoading.hidden = true;
    const canvas = document.createElement("canvas");
    canvas.id = "pdfCanvas";
    canvas.className = "pdf-page-canvas";
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "PDF 首页预览");
    el.pdfCanvas.replaceWith(canvas);
    el.pdfCanvas = canvas;
    return preview.generation;
  }

  async function renderPreview(url) {
    if (preview.url === url && ["loading", "rendering", "ready"].includes(preview.status)) return;
    const generation = destroyPreview();
    preview.url = url;
    preview.status = "loading";
    el.pdfPreview.hidden = true;
    el.previewEmpty.hidden = true;
    el.previewLoading.hidden = false;
    text(el.previewLoading, "正在读取 PDF 首页…");
    try {
      if (!pdfModulePromise) {
        pdfModulePromise = import("/vendor/pdf.mjs").catch((error) => { pdfModulePromise = null; throw error; });
      }
      const pdfjs = await pdfModulePromise;
      if (generation !== preview.generation) return;
      pdfjs.GlobalWorkerOptions.workerSrc = "/vendor/pdf.worker.mjs";
      const loadingTask = pdfjs.getDocument({
        url, rangeChunkSize: 65536, disableRange: false,
        // Prevent background downloading of every page. Result URLs remain
        // direct Range-capable HTTP requests rather than full-file blobs.
        disableAutoFetch: true, disableStream: true, useSystemFonts: true, isEvalSupported: false,
      });
      preview.loadingTask = loadingTask;
      loadingTask.onProgress = ({ loaded, total }) => {
        if (generation === preview.generation && preview.status === "loading" && total > 0) {
          text(el.previewLoading, `正在读取 PDF 首页… ${sizeText(loaded)} / ${sizeText(total)}`);
        }
      };
      const pdf = await loadingTask.promise;
      if (generation !== preview.generation) return;
      const page = await pdf.getPage(1);
      if (generation !== preview.generation) return;
      preview.document = pdf;
      preview.page = page;
      await renderPreviewPage(generation);
    } catch (error) {
      if (generation !== preview.generation || error.name === "RenderingCancelledException") return;
      preview.status = "error";
      el.previewLoading.hidden = true;
      el.pdfPreview.hidden = true;
      el.previewEmpty.hidden = false;
      el.loadPreview.hidden = false;
      text(el.previewEmptyTitle, "PDF 预览暂时无法载入");
      text(el.previewEmptyDetail, "可以点击重试；已完成的 PDF 仍可直接下载。");
      text(el.loadPreview, "重新加载预览");
    }
  }

  async function renderPreviewPage(generation = preview.generation) {
    const page = preview.page;
    if (!page || generation !== preview.generation) return;
    preview.status = "rendering";
    preview.renderTask?.cancel();
    // A fresh canvas prevents a cancelled render from sharing a drawing
    // context with its replacement during rapid tab switches or resizing.
    const canvas = document.createElement("canvas");
    canvas.id = "pdfCanvas";
    canvas.className = "pdf-page-canvas";
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", `${state.view === "original" ? "原稿" : "处理结果"} PDF 首页预览`);
    el.pdfCanvas.replaceWith(canvas);
    el.pdfCanvas = canvas;
    const natural = page.getViewport({ scale: 1 });
    const availableWidth = Math.max(64, el.previewStage.clientWidth - 48);
    const availableHeight = Math.max(64, el.previewStage.clientHeight - 48);
    const scale = Math.min(availableWidth / natural.width, availableHeight / natural.height);
    const viewport = page.getViewport({ scale });
    const dpr = Math.min(window.devicePixelRatio || 1, 2, Math.sqrt(4_000_000 / (viewport.width * viewport.height)));
    canvas.width = Math.max(1, Math.round(viewport.width * dpr));
    canvas.height = Math.max(1, Math.round(viewport.height * dpr));
    canvas.style.width = `${viewport.width}px`;
    canvas.style.height = `${viewport.height}px`;
    el.pdfPreview.hidden = false;
    text(el.previewLoading, "正在绘制 PDF 首页…");
    el.previewLoading.hidden = false;
    const renderTask = page.render({ canvasContext: canvas.getContext("2d"), viewport,
      transform: dpr === 1 ? null : [dpr, 0, 0, dpr, 0, 0], background: "#ffffff" });
    preview.renderTask = renderTask;
    try {
      await renderTask.promise;
      if (generation !== preview.generation || preview.renderTask !== renderTask) return;
      preview.status = "ready";
      el.previewLoading.hidden = true;
      text(el.previewPageInfo, `第 1 页 / 共 ${preview.document.numPages} 页`);
    } catch (error) {
      if (error.name !== "RenderingCancelledException") throw error;
    }
  }

  function setPreview(view = state.view) {
    state.view = view;
    const isOriginal = view === "original";
    const url = isOriginal ? state.originalUrl : state.resultUrl;
    const bytes = isOriginal ? state.pdf?.size || 0 : state.resultBytes;
    const approved = isOriginal ? state.originalApproved : state.resultApproved;
    const canLoad = Boolean(url) && (bytes <= MANUAL_PREVIEW_BYTES || approved);
    el.originalTab.setAttribute("aria-selected", String(isOriginal));
    el.resultTab.setAttribute("aria-selected", String(!isOriginal));
    text(el.previewFilename, isOriginal ? state.pdf?.name || "尚未选择稿件" : state.record?.original_filename || "处理后的 CMYK PDF");
    text(el.previewKind, isOriginal ? "原稿 PDF" : "DeviceCMYK PDF");
    el.loadPreview.hidden = !url || canLoad;
    text(el.loadPreview, "加载 PDF 预览");
    if (canLoad) {
      renderPreview(url);
    } else {
      destroyPreview();
      el.previewEmpty.hidden = false;
      el.pdfPreview.hidden = true;
      text(el.previewEmptyTitle, url ? "大幅面稿件，按需加载预览" : "添加 PDF，预览从这里开始");
      text(el.previewEmptyDetail, url ? `文件大小 ${sizeText(bytes)}，点击后按需读取并预览首页。` : "保留稿件尺寸、图层位置与透明关系。");
    }
  }

  function clearResult() {
    state.resultUrl = null;
    state.resultBytes = 0;
    state.resultApproved = false;
    state.record = null;
    el.resultMetrics.hidden = true;
    el.downloadPdf.hidden = true;
    el.downloadPdf.removeAttribute("href");
    el.retryPoll.hidden = true;
    setPreview("original");
  }

  function selectFile(kind, file, sample = false) {
    if (state.busy) return;
    showError("");
    if (!sample) el.sampleNotice.hidden = true;
    if (file) {
      const valid = kind === "pdf" ? /\.pdf$/i.test(file.name) || file.type === "application/pdf" : /\.csv$/i.test(file.name);
      if (!valid) { showError(kind === "pdf" ? "请选择 PDF 文件。" : "请选择 CMYK–Lab CSV 文件。"); return; }
      if (!file.size) { showError("文件内容为空，请重新选择。"); return; }
    }
    if (kind === "pdf" && state.originalUrl) URL.revokeObjectURL(state.originalUrl);
    state[kind] = file || null;
    if (kind === "pdf") {
      state.originalUrl = file ? URL.createObjectURL(file) : null;
      state.originalApproved = false;
    }
    const name = kind === "pdf" ? el.pdfName : el.csvName;
    const meta = kind === "pdf" ? el.pdfMeta : el.csvMeta;
    const drop = kind === "pdf" ? el.pdfDrop : el.csvDrop;
    text(name, file?.name || (kind === "pdf" ? "选择 PDF 稿件" : "选择 CMYK–Lab 色表"));
    text(meta, file ? `${sizeText(file.size)} · ${kind.toUpperCase()} 文件` : kind === "pdf" ? "点击选择，或拖放文件至此" : "同一组布料与设备的测量数据");
    drop.classList.toggle("has-file", Boolean(file));
    if (kind === "pdf") setPreview("original");
    updateControls();
  }

  function configureDrop(kind) {
    const input = kind === "pdf" ? el.pdfFile : el.csvFile;
    const drop = kind === "pdf" ? el.pdfDrop : el.csvDrop;
    input.addEventListener("change", () => selectFile(kind, input.files[0]));
    drop.addEventListener("dragover", (event) => { event.preventDefault(); if (!state.busy) drop.classList.add("drag-over"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("drag-over"));
    drop.addEventListener("drop", (event) => {
      event.preventDefault();
      drop.classList.remove("drag-over");
      if (state.busy) return;
      const files = event.dataTransfer?.files;
      if (files?.length !== 1) { showError("请每次拖入一个文件。"); return; }
      const transfer = new DataTransfer();
      transfer.items.add(files[0]);
      input.files = transfer.files;
      selectFile(kind, files[0]);
    });
  }

  function elapsedText(record) {
    const start = Date.parse(record.created_at) || state.startedAt;
    const end = Date.parse(record.updated_at) || Date.now();
    const seconds = Math.max(0, (end - (start || end)) / 1000);
    const rounded = Math.round(seconds);
    return seconds < 60 ? `${round(seconds)} 秒` : `${Math.floor(rounded / 60)} 分 ${rounded % 60} 秒`;
  }

  function applyRecord(record) {
    state.record = record;
    state.phase = record.status;
    state.busy = !TERMINAL.has(record.status);
    text(el.jobId, `JOB ${state.jobId}`);
    el.retryPoll.hidden = true;
    const progress = Math.max(0, Math.min(1, Number(record.progress) || 0));
    if (record.status === "succeeded") {
      const result = record.result || {};
      const info = result.info || {};
      state.resultBytes = Number(record.result_bytes) || 0;
      state.resultUrl = `/api/jobs/${encodeURIComponent(state.jobId)}/result?preview=1`;
      text(el.taskHeading, "颜色处理已完成");
      text(el.taskMessage, "CMYK PDF 已生成。页面尺寸与未选图层已保留，可预览或下载结果。");
      setProgress(100, 3);
      text(el.resultDevice, info.deviceName || state.capabilities?.gpuName || "GPU");
      const dpi = Number(result.dpi);
      text(el.resultDensity, state.options?.densityMode === "custom"
        ? `${round(dpi || state.options.dpi)} DPI` : Number.isFinite(dpi) && dpi > 0 ? `原稿 · ${round(dpi)} DPI` : "原稿密度");
      text(el.resultTime, elapsedText(record));
      el.resultMetrics.hidden = false;
      el.downloadPdf.href = `/api/jobs/${encodeURIComponent(state.jobId)}/result`;
      el.downloadPdf.download = "";
      el.downloadPdf.hidden = false;
      setPreview("result");
    } else if (record.status === "failed") {
      text(el.taskHeading, "处理未完成");
      text(el.taskMessage, record.error || "服务端处理失败，请检查 PDF 图层、CSV 数据与 GPU 状态后重试。");
      showError(record.error || "任务处理失败。");
      setProgress(10 + progress * 85, 2, true);
    } else if (record.status === "cancelled") {
      text(el.taskHeading, "任务已取消");
      text(el.taskMessage, "服务已停止此任务，可以调整设置后重新开始。");
      setProgress(10 + progress * 85, 2);
    } else if (record.status === "queued" || record.status === "uploading") {
      text(el.taskHeading, state.cancelRequested ? "正在取消任务" : "已上传，等待 GPU");
      text(el.taskMessage, state.cancelRequested ? "取消请求已发送，等待服务确认。" : "文件已接收，任务将按顺序进入 GPU 处理。");
      setProgress(10, 1);
    } else {
      text(el.taskHeading, state.cancelRequested ? "正在取消任务" : record.stage || "GPU 正在处理");
      text(el.taskMessage, state.cancelRequested ? "正在等待当前计算步骤结束，任务取消后会自动更新。"
        : record.message || (progress <= .03 ? "正在建立精细颜色引擎，首次运行可能需要编译 GPU 内核。" : "正在处理指定图层，完成后自动提供 PDF。"));
      setProgress(10 + progress * 85, 2);
    }
    if (!state.busy) state.cancelRequested = false;
    saveCurrentJob();
    updateControls();
  }

  function stopPolling() {
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }

  async function pollJob(generation = state.generation) {
    stopPolling();
    const sequence = ++state.pollSequence;
    const id = state.jobId;
    if (!id) return;
    try {
      const record = await fetchJson(`/api/jobs/${encodeURIComponent(id)}`);
      if (generation !== state.generation || sequence !== state.pollSequence || id !== state.jobId) return;
      applyRecord(record);
      if (!TERMINAL.has(record.status)) state.pollTimer = setTimeout(() => pollJob(generation), 1000);
    } catch (error) {
      if (generation !== state.generation || sequence !== state.pollSequence || id !== state.jobId) return;
      if (error.status === 404) {
        state.busy = false;
        state.phase = "failed";
        text(el.taskHeading, "任务已不存在");
        text(el.taskMessage, "任务可能已过保留期。请重新选择文件并创建任务。");
        state.jobId = null;
        saveCurrentJob();
      } else {
        text(el.taskHeading, "暂时无法查询进度");
        text(el.taskMessage, `${error.message} 任务可能仍在服务端运行，点击“继续查询”恢复连接。`);
        el.retryPoll.hidden = false;
      }
      updateControls();
    }
  }

  function failUpload(message) {
    state.xhr = null;
    state.busy = false;
    state.phase = "failed";
    text(el.taskHeading, "请求未完成");
    text(el.taskMessage, message);
    showError(message);
    setProgress(0, 0, true);
    updateControls();
  }

  function submit(event) {
    event.preventDefault();
    if (state.busy) return;
    showError("");
    let options;
    try {
      if (!state.gpuReady) throw new Error("GPU 尚未就绪，请先重新连接服务。");
      if (!state.pdf || !state.csv) throw new Error("请先选择 PDF 稿件与 CMYK–Lab CSV 色表。");
      options = readOptions();
      const limits = state.capabilities?.limits || {};
      if (limits.maxUploadBytes && state.pdf.size > limits.maxUploadBytes) throw new Error(`PDF 超过服务上限 ${sizeText(limits.maxUploadBytes)}。`);
      if (limits.maxCsvBytes && state.csv.size > limits.maxCsvBytes) throw new Error(`CSV 超过服务上限 ${sizeText(limits.maxCsvBytes)}。`);
    } catch (error) { showError(error.message); return; }
    stopPolling();
    state.generation += 1;
    const generation = state.generation;
    state.jobId = null;
    state.options = options;
    state.startedAt = Date.now();
    state.cancelRequested = false;
    clearResult();
    state.phase = "uploading";
    state.busy = true;
    saveCurrentJob();
    text(el.taskHeading, "正在上传文件");
    text(el.taskMessage, "文件通过当前 Demo 转发到 API，上传完成后进入 GPU 队列。");
    text(el.jobId, "正在等待服务分配任务编号");
    setProgress(0, 0);
    updateControls();
    const form = new FormData();
    form.append("pdf", state.pdf);
    form.append("csv", state.csv);
    form.append("layers", JSON.stringify(options.layers));
    if (options.densityMode === "custom") form.append("dpi", String(options.dpi));
    const xhr = new XMLHttpRequest();
    state.xhr = xhr;
    xhr.open("POST", `/api/${options.method}?wait=false`);
    xhr.upload.addEventListener("progress", (upload) => {
      if (generation !== state.generation || !upload.lengthComputable) return;
      const percent = upload.total ? upload.loaded / upload.total * 100 : 0;
      setProgress(percent * .1, 0);
      text(el.taskMessage, percent >= 100 ? "文件已发送，正在等待 API 检查文件与图层。" : `正在上传 ${Math.round(percent)}% · ${sizeText(upload.loaded)} / ${sizeText(upload.total)}`);
    });
    xhr.addEventListener("load", () => {
      if (generation !== state.generation) return;
      state.xhr = null;
      let payload;
      try { payload = JSON.parse(xhr.responseText); } catch { failUpload(`API 返回了无法识别的响应（HTTP ${xhr.status}）。`); return; }
      if (xhr.status !== 202 || !/^[a-f0-9]{32}$/i.test(payload.id || "")) {
        failUpload(friendlyError(payload, xhr.status === 429 ? "GPU 任务队列已满，请稍后重试。" : `请求失败（HTTP ${xhr.status}）。`));
        return;
      }
      state.jobId = payload.id;
      saveCurrentJob();
      applyRecord(payload);
      pollJob(generation);
    });
    xhr.addEventListener("error", () => { if (generation === state.generation) failUpload("上传连接中断，未收到任务编号。请检查 API 服务与网络后重试。"); });
    xhr.addEventListener("abort", () => {
      if (generation !== state.generation) return;
      state.xhr = null;
      state.busy = false;
      state.phase = "cancelled";
      state.cancelRequested = false;
      text(el.taskHeading, "上传已取消");
      text(el.taskMessage, "请求已停止，尚未收到服务端任务编号。");
      setProgress(0, 0);
      updateControls();
    });
    xhr.send(form);
  }

  async function cancelJob() {
    if (!state.busy || state.cancelRequested) return;
    if (state.xhr) { state.xhr.abort(); return; }
    if (!state.jobId) return;
    const id = state.jobId;
    state.cancelRequested = true;
    updateControls();
    text(el.taskHeading, "正在取消任务");
    text(el.taskMessage, "取消请求已发送，正在等待当前计算步骤结束。");
    try {
      const record = await fetchJson(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
      if (id !== state.jobId) return;
      applyRecord(record);
      pollJob();
    } catch (error) {
      if (id !== state.jobId) return;
      state.cancelRequested = false;
      text(el.taskMessage, `取消请求尚未确认：${error.message} 将继续查询任务状态。`);
      updateControls();
      pollJob();
    }
  }

  async function connect() {
    if (state.connecting) return;
    state.connecting = true;
    state.gpuReady = false;
    el.refreshConnection.disabled = true;
    el.connectionBadge.className = "connection";
    el.serviceDot.className = "service-dot";
    text(el.connectionText, "正在连接服务");
    updateControls();
    try {
      const config = await fetchJson("/demo-config");
      const apiUrl = new URL(config.apiBaseUrl);
      if (!["http:", "https:"].includes(apiUrl.protocol)) throw new Error("API 地址格式无效。");
      state.apiBaseUrl = apiUrl.href.replace(/\/$/, "");
      el.docsLink.href = `${state.apiBaseUrl}/docs/`;
      text(el.apiAddress, state.apiBaseUrl);
      const storageKey = `${STORAGE_PREFIX}${encodeURIComponent(state.apiBaseUrl)}`;
      if (state.storageKey !== storageKey) {
        if (state.storageKey) {
          // A changed proxy destination must not poll or download an old
          // service's job ID. Its saved entry remains in its own scope.
          stopPolling();
          state.generation += 1;
          state.xhr?.abort();
          state.xhr = null;
          state.jobId = null;
          state.options = null;
          state.busy = false;
          state.phase = "idle";
          state.cancelRequested = false;
          clearResult();
          text(el.taskHeading, "等待开始");
          text(el.taskMessage, "API 地址已更新，可以创建任务。");
          text(el.jobId, "所有处理在所连接的 API 服务完成");
          setProgress(0, -1);
        }
        state.storageKey = storageKey;
        restore();
      }
      const capabilities = await fetchJson("/api/capabilities");
      state.capabilities = capabilities;
      state.gpuReady = capabilities.gpuReady === true;
      el.connectionBadge.className = `connection ${state.gpuReady ? "connected" : "disconnected"}`;
      el.serviceDot.className = `service-dot ${state.gpuReady ? "ready" : "error"}`;
      text(el.connectionText, state.gpuReady ? "GPU 服务已连接" : "GPU 尚未就绪");
      const memory = Number(capabilities.compute?.gpu_memory_bytes);
      text(el.serviceDetails, state.gpuReady
        ? `${capabilities.gpuName || "NVIDIA GPU"}${memory > 0 ? ` · ${(memory / 1024 ** 3).toFixed(1)} GiB` : ""} · 精细处理`
        : capabilities.compute?.note || "API 已连接，GPU 运行环境尚未就绪。");
    } catch (error) {
      el.connectionBadge.className = "connection disconnected";
      el.serviceDot.className = "service-dot error";
      text(el.connectionText, "服务连接失败");
      text(el.serviceDetails, error.message);
    } finally {
      state.connecting = false;
      el.refreshConnection.disabled = false;
      updateControls();
    }
  }

  function resetJob() {
    if (state.busy) return;
    stopPolling();
    state.generation += 1;
    state.jobId = null;
    state.options = null;
    state.phase = "idle";
    state.cancelRequested = false;
    el.jobForm.reset();
    selectFile("pdf", null);
    selectFile("csv", null);
    clearResult();
    el.sampleNotice.hidden = true;
    showError("");
    text(el.taskHeading, "等待开始");
    text(el.taskMessage, "选择文件并设置图层后，即可开始处理。");
    text(el.jobId, "所有处理在所连接的 API 服务完成");
    setProgress(0, -1);
    try { if (state.storageKey) localStorage.removeItem(state.storageKey); } catch { /* Optional persistence. */ }
    updateControls();
  }

  function syntheticPdf() {
    const encode = (value) => new TextEncoder().encode(value);
    const image = new Uint8Array(128 * 64 * 3);
    for (let y = 0; y < 64; y += 1) for (let x = 0; x < 128; x += 1) {
      const offset = (y * 128 + x) * 3;
      image[offset] = Math.round(35 + x / 127 * 195);
      image[offset + 1] = Math.round(190 - y / 63 * 140);
      image[offset + 2] = Math.round(65 + Math.sin(x / 16) * 35 + y / 63 * 35);
    }
    const stream = (dictionary, bytes) => [encode(`<< ${dictionary} /Length ${bytes.length} >>\nstream\n`), bytes, encode("\nendstream")];
    const objects = [null,
      [encode("<< /Type /Catalog /Pages 2 0 R /OCProperties << /OCGs [5 0 R 6 0 R] /D << /Order [5 0 R 6 0 R] /ON [5 0 R 6 0 R] >> >> >>")],
      [encode("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")],
      [encode("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 320 200] /Resources << /XObject << /Im0 4 0 R >> /Properties << /Artwork 5 0 R /Technical 6 0 R >> /Font << /F1 9 0 R >> >> /Contents [7 0 R 8 0 R] >>")],
      stream("/Type /XObject /Subtype /Image /Width 128 /Height 64 /ColorSpace /DeviceRGB /BitsPerComponent 8", image),
      [encode("<< /Type /OCG /Name (Print_Artwork) >>")],
      [encode("<< /Type /OCG /Name (Technical) >>")],
      stream("", encode("/OC /Artwork BDC\nq 256 0 0 128 32 36 cm /Im0 Do Q\nEMC\n")),
      stream("", encode("/OC /Technical BDC\nq 0.15 0.22 0.17 RG 0.6 w 25 29 270 142 re S\n25 36 m 18 36 l S 32 29 m 32 22 l S 288 171 m 288 178 l S 295 164 m 302 164 l S Q\nBT /F1 7 Tf 0.3 g 32 182 Td (SYNTHETIC COLOR API SAMPLE) Tj 0 -166 Td (RGB texture / 36 DPI / technical guide preserved) Tj ET\nEMC\n")),
      [encode("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")],
    ];
    const chunks = [encode("%PDF-1.7\n"), new Uint8Array([37, 226, 227, 207, 211, 10])];
    const offsets = [0];
    let length = chunks.reduce((total, chunk) => total + chunk.length, 0);
    for (let id = 1; id < objects.length; id += 1) {
      offsets[id] = length;
      const parts = [encode(`${id} 0 obj\n`), ...objects[id], encode("\nendobj\n")];
      chunks.push(...parts);
      length += parts.reduce((total, part) => total + part.length, 0);
    }
    const xref = length;
    chunks.push(encode(`xref\n0 ${objects.length}\n0000000000 65535 f \n${offsets.slice(1).map((offset) => `${String(offset).padStart(10, "0")} 00000 n \n`).join("")}trailer\n<< /Size ${objects.length} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`));
    return new File(chunks, "synthetic-artwork-36dpi.pdf", { type: "application/pdf" });
  }

  function loadSample() {
    if (state.busy) return;
    const pdf = syntheticPdf();
    const rows = ["C,M,Y,K,L,a,b"];
    for (const c of [0, 100]) for (const m of [0, 100]) for (const y of [0, 100]) for (const k of [0, 100]) {
      rows.push([c, m, y, k, 100 - .2 * c - .15 * m - .1 * y - .4 * k, .4 * m - .2 * c, .4 * y - .1 * m].join(","));
    }
    const csv = new File(["\ufeff", rows.join("\n"), "\n"], "synthetic-cmyk-lab-16-points.csv", { type: "text/csv" });
    for (const [input, file] of [[el.pdfFile, pdf], [el.csvFile, csv]]) {
      const transfer = new DataTransfer();
      transfer.items.add(file);
      input.files = transfer.files;
    }
    selectFile("pdf", pdf, true);
    selectFile("csv", csv, true);
    el.layers.value = "Print_Artwork";
    el.densityMode.value = "original";
    el.sampleNotice.hidden = false;
    updateControls();
  }

  function restore() {
    if (!state.storageKey) return;
    let saved;
    try { saved = JSON.parse(localStorage.getItem(state.storageKey) || "null"); } catch { return; }
    const options = saved?.options;
    if (options && ["nearest", "predict"].includes(options.method)) {
      document.querySelector(`input[name="method"][value="${options.method}"]`).checked = true;
      if (Array.isArray(options.layers) && options.layers.every((name) => typeof name === "string")) el.layers.value = options.layers.join("\n");
      if (["original", "custom"].includes(options.densityMode)) el.densityMode.value = options.densityMode;
      if (Number.isFinite(Number(options.dpi)) && Number(options.dpi) > 0) el.dpi.value = String(options.dpi);
      state.options = options;
    }
    if (/^[a-f0-9]{32}$/i.test(saved?.jobId || "")) {
      state.jobId = saved.jobId;
      state.busy = true;
      state.phase = "queued";
      text(el.jobId, `JOB ${state.jobId}`);
      text(el.taskHeading, "正在恢复任务");
      text(el.taskMessage, "正在查询上次任务。原稿文件不会保存在浏览器中，完成后仍可下载结果。");
      pollJob();
    }
  }

  configureDrop("pdf");
  configureDrop("csv");
  el.jobForm.addEventListener("submit", submit);
  el.loadSample.addEventListener("click", loadSample);
  el.cancelJob.addEventListener("click", cancelJob);
  el.newJob.addEventListener("click", resetJob);
  el.refreshConnection.addEventListener("click", connect);
  el.retryPoll.addEventListener("click", () => { el.retryPoll.hidden = true; text(el.taskMessage, "正在重新查询任务状态…"); pollJob(); });
  el.originalTab.addEventListener("click", () => setPreview("original"));
  el.resultTab.addEventListener("click", () => setPreview("result"));
  el.loadPreview.addEventListener("click", () => {
    if (state.view === "original") state.originalApproved = true; else state.resultApproved = true;
    setPreview();
  });
  el.jobForm.addEventListener("input", () => { showError(""); updateControls(); if (!state.busy) { if (!state.jobId) state.options = null; saveCurrentJob(); } });
  el.jobForm.addEventListener("change", () => { updateControls(); if (!state.busy) { if (!state.jobId) state.options = null; saveCurrentJob(); } });
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (preview.page && !el.pdfPreview.hidden) {
          renderPreviewPage().catch(() => {
            const url = preview.url;
            preview.status = "error";
            if (url) renderPreview(url);
          });
        }
      }, 120);
    }).observe(el.previewStage);
  }
  window.addEventListener("beforeunload", () => { destroyPreview(); if (state.originalUrl) URL.revokeObjectURL(state.originalUrl); });
  updateControls();
  connect();
})();
