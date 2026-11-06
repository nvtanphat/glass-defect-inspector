// Workflow screens use only persisted API records. No seed metrics or demo history.
const workspaceScreen = document.getElementById('workspace-screen');
const workflowScreen = document.getElementById('workflow-screen');
let activeScreen = 'workspace';
let historyPage = 1;
let reviewedInspection = null;
let historyFilters = {};
let workflowObjectUrls = [];
let workflowEpoch = 0;

function node(tag, text, parent, attrs = {}) {
  const el = document.createElement(tag);
  if (text != null) el.textContent = text;
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  if (parent) parent.appendChild(el);
  return el;
}

function button(label, parent, action, attrs = {}) {
  const el = node('button', label, parent, {type: 'button', ...attrs});
  el.addEventListener('click', () => Promise.resolve().then(action).catch(err => setStatus(err.message, true)));
  return el;
}

function renderBadge(conclusion, parent) {
  const badgeClass = conclusion === 'PASS' ? 'wf-badge-pass' : (conclusion === 'REJECT' ? 'wf-badge-reject' : 'wf-badge-review');
  const span = node('span', null, parent, {class: `wf-badge ${badgeClass}`});
  node('span', null, span, {class: 'wf-badge-dot'});
  node('span', qualityLabel(conclusion), span);
  return span;
}

function clearWorkflow() {
  workflowObjectUrls = workflowObjectUrls.filter(url => {
    if (url === appState.currentImage || url === appState.annotatedImage || url === appState.gradcamImage) return true;
    URL.revokeObjectURL(url);
    return false;
  });
  workflowScreen.replaceChildren();
}

async function imageUrl(url) {
  const response = await apiResponse(url);
  const objectUrl = URL.createObjectURL(await response.blob());
  workflowObjectUrls.push(objectUrl);
  return objectUrl;
}

function gradcamLabel(metadata) {
  if (metadata.flat_cam || !metadata.nonzero_gradient) return 'Grad-CAM không có tín hiệu rõ cho dự đoán này.';
  const target = metadata.target_kind === 'detection' ? 'Giải thích lỗi' : 'Chưa có lỗi vượt ngưỡng; giải thích ứng viên';
  const selected = metadata.detection_id ? ` #${metadata.detection_id}` : '';
  return `Grad-CAM · ${target}${selected}: ${defectLabel(metadata.target_class)}. Màu nóng = ảnh hưởng lớn hơn; không phải ranh giới lỗi.`;
}

async function loadGradcam(imageId, detectionId = null) {
  const query = new URLSearchParams();
  if (detectionId) query.set('detection_id', detectionId);
  const qStr = query.toString() ? `?${query.toString()}` : '';
  const metadata = await fetchFromApi(`/api/images/${encodeURIComponent(imageId)}/gradcam${qStr}`, {method: 'POST'});
  return {metadata, url: await imageUrl(metadata.image_url),
    detail: metadata.detail_url ? await imageUrl(metadata.detail_url) : null};
}

function currentExplanationKey() {
  return `${appState.imageId}:${appState.selectedDefectId || 'auto'}`;
}

function resetExplanationView() {
  const control = document.getElementById('btn-gradcam');
  if (control.getAttribute('aria-pressed') === 'true') viewportImg.src = appState.currentImage;
  appState.gradcamImage = null;
  appState.gradcamMetadata = null;
  control.setAttribute('aria-pressed', 'false');
  control.textContent = 'Xem vùng chú ý';
  control.disabled = !appState.imageId;
  const selected = appState.detections.find(d => d.id === appState.selectedDefectId);
  if (selected?.crop_b64) roiThumbnail.src = selected.crop_b64;
  document.getElementById('gradcam-caption').textContent = 'Chọn lỗi trong bảng, rồi xem vùng ảnh hưởng đến đúng lỗi đó.';
}

document.getElementById('btn-gradcam').addEventListener('click', async () => {
  const control = document.getElementById('btn-gradcam');
  const caption = document.getElementById('gradcam-caption');
  const imageId = appState.imageId;
  const key = currentExplanationKey();
  if (!imageId) return;
  if (control.getAttribute('aria-pressed') === 'true') {
    viewportImg.src = appState.currentImage;
    control.setAttribute('aria-pressed', 'false');
    control.textContent = 'Xem vùng chú ý';
    caption.textContent = 'Đang xem ảnh gốc và vị trí lỗi.';
    const selected = appState.detections.find(d => d.id === appState.selectedDefectId);
    if (selected?.crop_b64) roiThumbnail.src = selected.crop_b64;
    return;
  }
  control.disabled = true;
  control.textContent = 'Đang tạo bản đồ…';
  caption.textContent = 'Đang tính vùng chú ý từ model đã kiểm tra ảnh này.';
  try {
    if (!appState.gradcamImage) {
      const loaded = await loadGradcam(imageId, appState.selectedDefectId);
      if (key !== currentExplanationKey()) return;
      appState.gradcamImage = loaded.url;
      appState.gradcamMetadata = loaded.metadata;
      appState.gradcamDetail = loaded.detail;
    }
    viewportImg.src = appState.gradcamImage;
    control.setAttribute('aria-pressed', 'true');
    control.textContent = 'Xem ảnh gốc';
    caption.textContent = gradcamLabel(appState.gradcamMetadata);
    if (appState.gradcamDetail) roiThumbnail.src = appState.gradcamDetail;
  } catch (error) {
    if (key !== currentExplanationKey()) return;
    caption.textContent = 'Chưa tạo được bản đồ chú ý. Bạn có thể thử lại.';
    control.textContent = 'Thử lại';
    setStatus(error.message, true);
  } finally {
    if (key === currentExplanationKey()) control.disabled = false;
  }
});

async function exportInspection(id, format) {
  const response = await apiResponse(`/api/inspections/${encodeURIComponent(id)}/export?format=${format}`);
  const url = URL.createObjectURL(await response.blob());
  const anchor = node('a'); anchor.href = url; anchor.download = `inspection-${id}.${format}`;
  anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function showScreen(screen) {
  if (appState.isInspecting) { setStatus('Vui lòng chờ kiểm tra ảnh hoàn tất.'); return; }
  activeScreen = screen;
  const epoch = ++workflowEpoch;
  workspaceScreen.style.display = screen === 'workspace' ? '' : 'none';
  workflowScreen.hidden = screen === 'workspace';
  workflowScreen.classList.toggle('dashboard-mode', screen === 'dashboard');
  document.querySelectorAll('[data-screen]').forEach(el => el.classList.toggle('active', el.dataset.screen === screen));
  if (screen === 'workspace') { fitImage(); return; }
  clearWorkflow();
  node('p', 'Đang tải…', workflowScreen);
  if (screen === 'history') await loadHistory(epoch);
  if (screen === 'review') {
    if (reviewedInspection || appState.inspectionId) await openInspection(reviewedInspection || appState.inspectionId);
    else { clearWorkflow(); node('p', 'Chọn một kết quả trong Lịch sử kiểm tra để xác nhận.', workflowScreen); }
  }
  if (screen === 'dashboard') await loadDashboard(epoch);
}

document.querySelectorAll('[data-screen]').forEach(el => {
  el.addEventListener('click', () => showScreen(el.dataset.screen).catch(err => {
    clearWorkflow(); node('p', err.message, workflowScreen); setStatus(err.message, true);
  }));
});

async function loadHistory(epoch = ++workflowEpoch) {
  const params = new URLSearchParams({...historyFilters, page: historyPage, page_size: 10});
  const data = await fetchFromApi(`/api/inspections?${params}`);
  if (epoch !== workflowEpoch) return;
  clearWorkflow();

  // Header Title
  const headerDiv = node('div', null, workflowScreen, {style: 'display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;'});
  const titleBox = node('div', null, headerDiv);
  node('h2', 'Lịch sử kiểm tra chai', titleBox, {style: 'margin-bottom:4px;'});
  node('p', 'Tra cứu kết quả kiểm định, xem vị trí lỗi và thẩm định chất lượng chai thủy tinh.', titleBox, {style: 'font-size:12px;color:#64748b;'});
  const totalTag = node('span', null, headerDiv, {style: 'font-size:12px;font-weight:600;padding:6px 12px;background:#ffffff;border:1px solid #e2e8f0;border-radius:4px;box-shadow:0 1px 2px rgba(0,0,0,0.03);'});
  node('span', `Tổng cộng: ${data.total} lượt`, totalTag);

  // Search & Filter Card
  const controls = node('div', null, workflowScreen, {'data-card': '', style: 'display:flex;gap:10px;flex-wrap:wrap;align-items:center;padding:16px 20px;'});
  const q = node('input', null, controls, {placeholder: 'Tìm mã lượt kiểm tra (ID)...', 'aria-label': 'Tìm mã lượt kiểm tra', style: 'flex:1;min-width:200px;'});
  q.value = historyFilters.q || '';
  const status = node('select', null, controls, {'aria-label': 'Tình trạng xử lý'});
  for (const value of ['', 'PENDING', 'PROCESSING', 'COMPLETED', 'FAILED']) status.add(new Option(statusLabel(value) || 'Tất cả tình trạng', value));
  status.value = historyFilters.status || '';
  const conclusion = node('select', null, controls, {'aria-label': 'Kết luận'});
  for (const value of ['', 'PASS', 'REJECT', 'REVIEW']) conclusion.add(new Option(qualityLabel(value) || 'Tất cả kết luận', value));
  conclusion.value = historyFilters.conclusion || '';
  const model = node('select', null, controls, {'aria-label': 'Bộ kiểm tra'});
  model.add(new Option('Tất cả bộ kiểm tra', ''));
  for (const option of modelWeightsSelect.options) if (option.value) model.add(new Option(option.textContent, option.value));
  model.value = historyFilters.model_id || '';

  button('Tìm kiếm', controls, async () => {
    historyFilters = {};
    for (const [key, value] of Object.entries({q: q.value.trim(), status: status.value, conclusion: conclusion.value, model_id: model.value})) {
      if (value) historyFilters[key] = value;
    }
    historyPage = 1; await loadHistory();
  }, {class: 'wf-btn-primary'});

  if (Object.keys(historyFilters).length > 0) {
    button('Xóa lọc', controls, async () => {
      historyFilters = {};
      historyPage = 1;
      await loadHistory();
    });
  }

  // Data Table Card
  const tableCard = node('div', null, workflowScreen, {'data-card': '', style: 'padding:0;overflow:hidden;'});
  const table = node('table', null, tableCard);
  const header = node('tr', null, node('thead', null, table));
  for (const label of ['Mã lượt kiểm tra', 'Thời gian', 'Bộ kiểm tra', 'Tình trạng', 'Kết luận', 'Thao tác']) node('th', label, header);
  const body = node('tbody', null, table);
  for (const item of data.items) {
    const row = node('tr', null, body);
    // 1. Mã lượt kiểm tra
    const tdId = node('td', null, row);
    node('code', item.id, tdId, {style: 'font-family:monospace;font-size:11px;font-weight:700;color:#2563eb;background:#eff6ff;padding:3px 8px;border-radius:4px;border:1px solid #dbeafe;'});
    // 2. Thời gian
    node('td', new Date(item.created_at).toLocaleString('vi-VN'), row, {style: 'font-size:12px;color:#475569;'});
    // 3. Bộ kiểm tra
    node('td', modelLabel(item.model_id), row, {style: 'font-weight:500;'});
    // 4. Tình trạng
    const tdStatus = node('td', null, row);
    const statusBadge = node('span', null, tdStatus, {class: 'wf-badge wf-badge-info'});
    node('span', null, statusBadge, {class: 'wf-badge-dot'});
    node('span', statusLabel(item.status), statusBadge);
    // 5. Kết luận
    const tdConclusion = node('td', null, row);
    renderBadge(item.conclusion, tdConclusion);
    // 6. Thao tác
    const tdAction = node('td', null, row);
    button('Xem và xác nhận', tdAction, () => openInspection(item.id), {'data-inspection-id': item.id, style: 'font-size:12px;padding:5px 12px;background:#eff6ff;color:#1d4ed8;border-color:#bfdbfe;font-weight:600;'});
  }

  if (!data.items.length) {
    const emptyBox = node('div', null, workflowScreen, {'data-card': '', style: 'text-align:center;padding:36px;color:#64748b;'});
    node('p', ' Không tìm thấy kết quả kiểm tra phù hợp. Hãy thử điều kiện tìm kiếm khác.', emptyBox, {style: 'font-size:14px;font-weight:500;'});
  }

  // Pagination Bar
  const totalPages = Math.ceil(data.total / data.page_size) || 1;
  const pager = node('div', null, workflowScreen, {style: 'display:flex;align-items:center;justify-content:space-between;margin-top:12px;padding:4px 8px;'});
  const previous = button('Trang trước', pager, async () => { historyPage--; await loadHistory(); });
  previous.disabled = historyPage === 1;
  node('span', `Trang ${data.page} / ${totalPages} · ${data.total} lượt kiểm tra`, pager, {style: 'font-size:12px;font-weight:600;color:#64748b;'});
  const next = button('Trang sau', pager, async () => { historyPage++; await loadHistory(); });
  next.disabled = historyPage * data.page_size >= data.total;
}

async function openInspection(id) {
  const epoch = ++workflowEpoch;
  const detail = await fetchFromApi(`/api/inspections/${encodeURIComponent(id)}`);
  if (epoch !== workflowEpoch) return;
  reviewedInspection = id;
  activeScreen = 'review';
  workspaceScreen.style.display = 'none'; workflowScreen.hidden = false;
  document.querySelectorAll('[data-screen]').forEach(el => el.classList.toggle('active', el.dataset.screen === 'review'));
  clearWorkflow();

  // Top navigation bar / breadcrumb
  const topNav = node('div', null, workflowScreen, {style: 'display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;'});
  button('Quay lại Lịch sử kiểm tra', topNav, () => showScreen('history'), {style: 'font-size:12px;font-weight:600;'});

  // Export Buttons Toolbar
  const exportToolbar = node('div', null, topNav, {style: 'display:flex;gap:6px;'});
  for (const format of ['json', 'csv', 'pdf']) {
    const labels = {json: 'Tải JSON', csv: 'Tải CSV', pdf: 'Tải báo cáo PDF'};
    button(labels[format], exportToolbar, () => exportInspection(id, format), {style: 'font-size:12px;font-weight:600;'});
  }

  // Main Detail Header Card
  const card = node('div', null, workflowScreen, {'data-card': ''});
  const titleRow = node('div', null, card, {style: 'display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:12px;'});
  node('h2', `Xác nhận kết quả — ${detail.id}`, titleRow, {style: 'margin:0;font-size:18px;font-weight:800;'});
  
  // Status & Conclusion Badge
  const badgeWrap = node('div', null, titleRow);
  renderBadge(detail.conclusion, badgeWrap);

  // Review Summary paragraph (Strictly preserve ID and text content for tests)
  node('p', `${statusLabel(detail.status)} / ${qualityLabel(detail.conclusion)} · ${modelLabel(detail.model_id)}`, card, {id: 'review-summary', style: 'font-size:13px;font-weight:600;color:#334155;margin-bottom:10px;'});

  // Assessment text alert callout
  const isPass = detail.conclusion === 'PASS';
  const isReject = detail.conclusion === 'REJECT';
  const calloutStyle = isPass
    ? 'background:#ecfdf5;border:1px solid #a7f3d0;color:#065f46;'
    : (isReject ? 'background:#fff1f2;border:1px solid #fecdd3;color:#9f1239;' : 'background:#fffbeb;border:1px solid #fde68a;color:#92400e;');
  const callout = node('div', null, card, {style: `padding:12px 16px;border-radius:4px;font-size:13px;font-weight:500;margin-bottom:14px;${calloutStyle}`});
  node('p', `${assessmentText(detail)}`, callout);

  // Metadata Grid
  const metaGrid = node('div', null, card, {style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;padding-top:12px;border-top:1px solid #f1f5f9;font-size:12px;color:#64748b;'});
  node('div', `Thời gian: ${new Date(detail.created_at).toLocaleString('vi-VN')}`, metaGrid);
  node('div', `Phiên bản: v${detail.version}`, metaGrid);
  node('div', `Số lượng ảnh: ${detail.images.length} ảnh`, metaGrid);

  // Images and Defect Cards
  for (const image of detail.images) {
    const imageCard = node('div', null, workflowScreen, {'data-card': ''});
    node('h3', `Ảnh kiểm định: ${image.filename}`, imageCard, {style: 'font-size:15px;font-weight:700;margin-bottom:8px;'});
    const src = await imageUrl(image.annotated_url || image.original_url);
    if (epoch !== workflowEpoch) return;

    const result = image.result;
    const summaryText = result ? `${result.total_defects} lỗi phát hiện · ${durationLabel(result.latency_ms)}` : 'Chưa có kết quả hoàn chỉnh';
    node('p', summaryText, imageCard, {style: 'font-size:12px;color:#475569;margin-bottom:14px;font-weight:500;'});

    // 2-column image and inspector details
    const splitGrid = node('div', null, imageCard, {style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px;align-items:start;'});
    
    // Left column: Bottle image preview
    const leftCol = node('div', null, splitGrid);
    node('img', null, leftCol, {src, alt: 'Ảnh kết quả đã lưu', style: 'max-height:420px;width:100%;object-fit:contain;background:#f8fafc;border:1px solid #e2e8f0;border-radius:4px;'});

    // Right column: Grad-CAM tools & Defect list
    const rightCol = node('div', null, splitGrid);

    if (result) {
      // Grad-CAM Explainability box
      const camBox = node('div', null, rightCol, {style: 'background:#f8fafc;border:1px solid #e2e8f0;border-radius:4px;padding:14px;margin-bottom:14px;'});
      node('h4', 'Giải thích quyết định AI (Grad-CAM)', camBox, {style: 'font-size:13px;font-weight:700;color:#0f172a;margin-bottom:8px;'});
      
      const camTarget = node('select', null, camBox, {'aria-label': 'Lỗi cần giải thích', style: 'width:100%;margin-bottom:8px;'});
      for (const d of result.detections) camTarget.add(new Option(`Lỗi #${d.id} · ${defectLabel(d.class_name)}`, d.id));
      if (!result.detections.length) camTarget.add(new Option('Ứng viên chưa vượt ngưỡng', ''));
      
      const camButton = button('Xem vùng chú ý', camBox, async () => {
        if (camImage.src) {
          camImage.hidden = !camImage.hidden;
          camButton.textContent = camImage.hidden ? 'Xem vùng chú ý' : 'Ẩn bản đồ';
          return;
        }
        camButton.disabled = true;
        camNote.textContent = 'Đang tính toán bản đồ chú ý Gradient…';
        const selection = `${camTarget.value}`;
        try {
          const loaded = await loadGradcam(image.id, camTarget.value || null);
          if (epoch !== workflowEpoch || selection !== `${camTarget.value}`) return;
          camImage.src = loaded.url;
          camImage.hidden = false;
          camButton.textContent = 'Ẩn bản đồ';
          camNote.textContent = gradcamLabel(loaded.metadata);
          const download = node('a', 'Tải ảnh Grad-CAM', camBox, {href: loaded.url, download: `gradcam-${image.id}.jpg`});
          download.style.cssText = 'display:inline-block;margin:8px 0;font-size:12px;font-weight:600;color:#1d4ed8;';
        } catch (error) {
          camNote.textContent = 'Chưa tạo được Grad-CAM. Bạn có thể thử lại.';
          camButton.textContent = 'Thử lại Grad-CAM';
          setStatus(error.message, true);
        } finally { camButton.disabled = false; }
      }, {style: 'font-size:12px;font-weight:600;padding:6px 12px;background:#eff6ff;color:#1d4ed8;border-color:#bfdbfe;'});

      const camNote = node('p', 'Xem vùng model chú ý cho ảnh đã kiểm tra.', camBox, {class: 'gradcam-note', role: 'status', style: 'font-size:11px;color:#64748b;margin-top:6px;'});
      const camImage = node('img', null, camBox, {alt: 'Bản đồ Grad-CAM trên ảnh kiểm định', hidden: ''});
      camImage.style.cssText = 'max-height:360px;width:100%;object-fit:contain;margin-top:8px;border-radius:4px;border:1px solid #cbd5e1;';

      const resetSavedCam = () => {
        camImage.removeAttribute('src'); camImage.hidden = true;
        camButton.textContent = 'Xem vùng chú ý'; camButton.disabled = false;
        camBox.querySelectorAll('a').forEach(link => link.remove());
      };
      camTarget.addEventListener('change', resetSavedCam);

      // Detection Table
      const detectionsCard = node('div', null, rightCol, {style: 'margin-bottom:14px;'});
      node('h4', `Danh sách lỗi phát hiện (${result.detections.length})`, detectionsCard, {style: 'font-size:13px;font-weight:700;color:#0f172a;margin-bottom:8px;'});
      const table = node('table', null, detectionsCard);
      const head = node('tr', null, node('thead', null, table));
      node('th', 'Loại lỗi', head);
      node('th', 'Độ tin cậy', head);
      const bdy = node('tbody', null, table);
      for (const d of result.detections) {
        const row = node('tr', null, bdy);
        node('td', defectLabel(d.class_name), row, {style: 'font-weight:600;'});
        node('td', `${(d.confidence * 100).toFixed(1)}%`, row, {style: 'color:#2563eb;font-weight:600;'});
      }
      if (!result.detections.length) {
        const emptyRow = node('tr', null, bdy);
        node('td', 'Không phát hiện lỗi vượt ngưỡng.', emptyRow, {colspan: '2', style: 'text-align:center;color:#64748b;font-size:12px;'});
      }

      // Action button to examine in workspace (Preserve exact name for test)
      button('Xem vị trí lỗi trên ảnh', rightCol, async () => {
        const epoch = workflowEpoch;
        const original = await imageUrl(image.original_url);
        if (epoch !== workflowEpoch) return;
        displayImage(original);
        appState.imageWidth = result.image_width; appState.imageHeight = result.image_height;
        handleInferenceResult({...result, success: true, inspection_id: id, image_id: image.id, conclusion: detail.conclusion, annotated_b64: src});
        await showScreen('workspace');
      }, {style: 'width:100%;padding:10px;font-size:13px;font-weight:700;background:#2563eb;color:#ffffff;border-color:#2563eb;'});
    }
  }

  // Operator / Technician Sign-off Form
  if (detail.status === 'COMPLETED') {
    const form = node('form', null, workflowScreen, {'data-card': '', id: 'review-form', style: 'border-left:4px solid #2563eb;'});
    node('h3', 'Xác nhận của người kiểm tra (Thẩm định QC)', form, {style: 'margin-bottom:4px;'});
    node('p', 'Kỹ thuật viên kiểm soát chất lượng xác nhận kết luận hoặc phân loại lại phế phẩm trước khi xuất xưởng.', form, {style: 'font-size:12px;color:#64748b;margin-bottom:16px;'});
    
    // Conclusion Select
    const field1 = node('div', null, form, {style: 'margin-bottom:12px;'});
    node('label', 'Kết luận sau khi kiểm tra:', field1, {style: 'display:block;font-size:12px;font-weight:700;color:#334155;margin-bottom:4px;'});
    const choice = node('select', null, field1, {id: 'review-conclusion', 'aria-label': 'Kết luận xác nhận', style: 'width:100%;max-width:320px;'});
    for (const value of ['PASS', 'REJECT', 'REVIEW']) choice.add(new Option(qualityLabel(value), value));
    choice.value = detail.conclusion;

    // Reviewer Name Input
    const field2 = node('div', null, form, {style: 'margin-bottom:12px;'});
    node('label', 'Tên kỹ thuật viên thẩm định:', field2, {style: 'display:block;font-size:12px;font-weight:700;color:#334155;margin-bottom:4px;'});
    const reviewer = node('input', null, field2, {id: 'reviewer', required: '', maxlength: '120', placeholder: 'Ví dụ: Kỹ thuật viên kiểm định 01', 'aria-label': 'Tên người kiểm tra', style: 'width:100%;max-width:400px;'});

    // Reason Textarea
    const field3 = node('div', null, form, {style: 'margin-bottom:16px;'});
    node('label', 'Lý do xác nhận hoặc thay đổi kết luận:', field3, {style: 'display:block;font-size:12px;font-weight:700;color:#334155;margin-bottom:4px;'});
    const reason = node('textarea', null, field3, {id: 'review-reason', required: '', minlength: '3', maxlength: '2000', placeholder: 'Ghi rõ cơ sở xác nhận hoặc loại bỏ sản phẩm...', 'aria-label': 'Lý do xác nhận', style: 'width:100%;height:80px;'});

    // Submit button
    const submit = node('button', 'Lưu xác nhận', form, {type: 'submit', id: 'submit-review'});
    form.addEventListener('submit', async event => {
      event.preventDefault(); submit.disabled = true;
      try {
        await fetchFromApi(`/api/inspections/${id}/reviews`, {method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({conclusion: choice.value, reviewer: reviewer.value, reason: reason.value, expected_version: detail.version})});
        setStatus('Đã lưu xác nhận'); await openInspection(id);
      } catch (err) { setStatus(err.message, true); submit.disabled = false; }
    });
  }

  // Audit Trail History
  const history = node('div', null, workflowScreen, {'data-card': ''});
  node('h3', 'Lịch sử xác nhận (Audit Trail)', history, {style: 'margin-bottom:10px;'});
  if (!detail.reviews.length) node('p', 'Chưa có người xác nhận kết quả này.', history, {style: 'font-size:13px;color:#64748b;'});
  for (const review of detail.reviews) {
    const revCard = node('div', null, history, {style: 'background:#f8fafc;border:1px solid #e2e8f0;border-radius:4px;padding:12px 14px;margin-bottom:8px;font-size:13px;'});
    node('p', `v${review.version} · ${review.reviewer} · ${qualityLabel(review.previous_conclusion)} → ${qualityLabel(review.conclusion)} · ${review.reason} · ${new Date(review.created_at).toLocaleString('vi-VN')}`, revCard, {style: 'color:#1e293b;font-weight:500;'});
  }
}

async function loadDashboard(epoch = ++workflowEpoch) {
  const data = await fetchFromApi('/api/dashboard');
  if (epoch !== workflowEpoch) return;
  clearWorkflow();
  workflowScreen.classList.add('dashboard-mode');

  // Header Title
  const headerDiv = node('div', null, workflowScreen, {style: 'display:flex;align-items:center;justify-content:space-between;margin-bottom:2px;flex-shrink:0;'});
  const titleBox = node('div', null, headerDiv);
  node('h2', 'Thống kê chất lượng & Hiệu suất AI', titleBox, {style: 'font-size:15px;font-weight:800;color:#0f172a;margin:0;'});
  node('p', 'Báo cáo tổng hợp số liệu kiểm định chai thủy tinh thời gian thực.', titleBox, {style: 'font-size:11px;color:#64748b;margin:0;'});
  const realtimeTag = node('span', null, headerDiv, {style: 'font-size:11px;font-weight:600;padding:3px 10px;background:#ffffff;border:1px solid #e2e8f0;border-radius:4px;color:#475569;'});
  node('span', `Cập nhật trực tiếp · ${data.total_inspections} ca`, realtimeTag);

  // Top KPI 4 Cards Grid
  const kpiGrid = node('div', null, workflowScreen, {class: 'wf-kpi-grid', style: 'display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:6px;flex-shrink:0;'});
  
  // KPI 1: Tổng lượt kiểm tra (Preserve id="dashboard-total" and "Tổng lượt kiểm tra:")
  const kpi1 = node('div', null, kpiGrid, {class: 'wf-kpi-card', style: 'padding:8px 12px;'});
  node('div', 'TỔNG LƯỢT KIỂM ĐỊNH', kpi1, {class: 'wf-kpi-header', style: 'font-size:10px;font-weight:700;color:#64748b;letter-spacing:0.05em;margin-bottom:2px;'});
  node('div', String(data.total_inspections), kpi1, {class: 'wf-kpi-val', style: 'font-size:20px;font-weight:800;color:#0f172a;line-height:1.1;'});
  node('p', `Tổng lượt kiểm tra: ${data.total_inspections} · Ảnh đã kiểm tra: ${data.completed_images}`, kpi1, {id: 'dashboard-total', class: 'wf-kpi-sub', style: 'font-size:11px;color:#64748b;margin-top:2px;'});

  // KPI 2: Tỷ lệ ĐẠT (Yield)
  const passCount = data.conclusions.PASS || 0;
  const passRate = data.total_inspections ? ((passCount / data.total_inspections) * 100).toFixed(1) : '0.0';
  const kpi2 = node('div', null, kpiGrid, {class: 'wf-kpi-card', style: 'padding:8px 12px;'});
  node('div', 'TỶ LỆ ĐẠT (YIELD)', kpi2, {class: 'wf-kpi-header', style: 'font-size:10px;font-weight:700;color:#059669;letter-spacing:0.05em;margin-bottom:2px;'});
  node('div', `${passRate}%`, kpi2, {class: 'wf-kpi-val', style: 'font-size:20px;font-weight:800;color:#059669;line-height:1.1;'});
  node('p', `${passCount} sản phẩm đạt chuẩn`, kpi2, {class: 'wf-kpi-sub', style: 'font-size:11px;color:#64748b;margin-top:2px;'});

  // KPI 3: Tỷ lệ PHẾ PHẨM (Reject Rate)
  const rejectCount = data.conclusions.REJECT || 0;
  const rejectRate = data.total_inspections ? ((rejectCount / data.total_inspections) * 100).toFixed(1) : '0.0';
  const kpi3 = node('div', null, kpiGrid, {class: 'wf-kpi-card', style: 'padding:8px 12px;'});
  node('div', 'TỶ LỆ PHẾ PHẨM (REJECT)', kpi3, {class: 'wf-kpi-header', style: 'font-size:10px;font-weight:700;color:#e11d48;letter-spacing:0.05em;margin-bottom:2px;'});
  node('div', `${rejectRate}%`, kpi3, {class: 'wf-kpi-val', style: 'font-size:20px;font-weight:800;color:#e11d48;line-height:1.1;'});
  node('p', `${rejectCount} chai phát hiện lỗi`, kpi3, {class: 'wf-kpi-sub', style: 'font-size:11px;color:#64748b;margin-top:2px;'});

  // KPI 4: Tốc độ kiểm tra AI
  const kpi4 = node('div', null, kpiGrid, {class: 'wf-kpi-card', style: 'padding:8px 12px;'});
  node('div', 'TỐC ĐỘ XỬ LÝ TRUNG BÌNH', kpi4, {class: 'wf-kpi-header', style: 'font-size:10px;font-weight:700;color:#64748b;letter-spacing:0.05em;margin-bottom:2px;'});
  node('div', durationLabel(data.mean_latency_ms), kpi4, {class: 'wf-kpi-val', style: 'font-size:20px;font-weight:800;color:#0f172a;line-height:1.1;'});
  node('p', 'Mô hình YOLO26m-Distilled', kpi4, {class: 'wf-kpi-sub', style: 'font-size:11px;color:#64748b;margin-top:2px;'});

  // 3-Column Analytics Grid (All in 1 Row side-by-side, fills remaining vertical height)
  const mainGrid = node('div', null, workflowScreen, {class: 'dashboard-main-grid', style: 'display:grid;grid-template-columns:1fr 1.25fr 1.35fr;gap:10px;flex:1;min-height:0;margin-bottom:0;'});

  // Card A: Kết luận chất lượng
  const conclusionsCard = node('div', null, mainGrid, {'data-card': '', style: 'padding:10px 14px;margin-bottom:0;display:flex;flex-direction:column;min-height:0;overflow:hidden;'});
  node('h3', 'Kết luận chất lượng', conclusionsCard, {style: 'font-size:13px;font-weight:700;color:#1e293b;margin:0 0 6px 0;flex-shrink:0;'});
  const concBody = node('div', null, conclusionsCard, {style: 'flex:1;min-height:0;overflow-y:auto;padding-right:2px;'});
  const maxConclusion = Math.max(...Object.values(data.conclusions), 1);
  const conclusionColors = {PASS: '#10b981', REJECT: '#f43f5e', REVIEW: '#f59e0b'};
  for (const [key, count] of Object.entries(data.conclusions)) {
    const row = node('div', null, concBody, {class: 'wf-metric-row', style: 'display:flex;justify-content:space-between;align-items:center;margin-top:6px;font-size:12px;font-weight:600;'});
    node('span', qualityLabel(key), row);
    node('span', `${count} (${((count / (data.total_inspections || 1)) * 100).toFixed(1)}%)`, row, {style: 'color:#64748b;font-size:11px;'});
    const track = node('div', null, concBody, {class: 'wf-prog-track', style: 'height:6px;background:#f1f5f9;border-radius:4px;overflow:hidden;margin:3px 0 8px 0;'});
    const fill = node('div', null, track, {class: 'wf-prog-fill', role: 'img', 'aria-label': `${qualityLabel(key)}: ${count}`});
    fill.style.cssText = `height:100%;border-radius:4px;width:${(count / maxConclusion) * 100}%;background:${conclusionColors[key] || '#2563eb'};`;
  }

  // Card B: Các loại lỗi (Defect Breakdown)
  const defectsCard = node('div', null, mainGrid, {'data-card': '', style: 'padding:10px 14px;margin-bottom:0;display:flex;flex-direction:column;min-height:0;overflow:hidden;'});
  node('h3', 'Các loại lỗi phát hiện', defectsCard, {style: 'font-size:13px;font-weight:700;color:#1e293b;margin:0 0 6px 0;flex-shrink:0;'});
  const defBody = node('div', null, defectsCard, {style: 'flex:1;min-height:0;overflow-y:auto;padding-right:2px;'});
  const maxDefect = Math.max(...Object.values(data.defect_classes), 1);
  const totalDefects = Object.values(data.defect_classes).reduce((a, b) => a + b, 0);
  for (const [key, count] of Object.entries(data.defect_classes)) {
    const row = node('div', null, defBody, {class: 'wf-metric-row', style: 'display:flex;justify-content:space-between;align-items:center;margin-top:4px;font-size:12px;font-weight:600;'});
    node('span', defectLabel(key), row);
    node('span', `${count} (${totalDefects ? ((count / totalDefects) * 100).toFixed(1) : 0}%)`, row, {style: 'color:#64748b;font-size:11px;'});
    const track = node('div', null, defBody, {class: 'wf-prog-track', style: 'height:6px;background:#f1f5f9;border-radius:4px;overflow:hidden;margin:2px 0 6px 0;'});
    const fill = node('div', null, track, {class: 'wf-prog-fill', role: 'img', 'aria-label': `${defectLabel(key)}: ${count}`});
    fill.style.cssText = `height:100%;border-radius:4px;width:${(count / maxDefect) * 100}%;background:#2563eb;`;
  }

  // Card C: Số lượt kiểm tra theo ngày (Trend Table)
  const trendCard = node('div', null, mainGrid, {'data-card': '', style: 'padding:10px 14px;margin-bottom:0;display:flex;flex-direction:column;min-height:0;overflow:hidden;'});
  node('h3', 'Số lượt kiểm tra theo ngày', trendCard, {style: 'font-size:13px;font-weight:700;color:#1e293b;margin:0 0 6px 0;flex-shrink:0;'});
  const trendBody = node('div', null, trendCard, {style: 'flex:1;min-height:0;overflow-y:auto;'});
  if (!data.trend.length) {
    node('p', 'Chưa có lượt kiểm tra hoàn thành.', trendBody, {style: 'color:#64748b;font-size:12px;margin-top:10px;'});
  } else {
    const table = node('table', null, trendBody, {style: 'width:100%;border-collapse:collapse;'});
    const header = node('tr', null, node('thead', null, table));
    for (const label of ['Ngày', 'Kết luận', 'Số lượt']) {
      node('th', label, header, {style: 'font-size:10px;padding:5px 8px;background:#f8fafc;font-weight:700;color:#475569;text-align:left;border-bottom:1px solid #e2e8f0;'});
    }
    const body = node('tbody', null, table);
    for (const item of data.trend) {
      const row = node('tr', null, body);
      node('td', item.date, row, {style: 'font-weight:600;font-size:11px;padding:5px 8px;border-bottom:1px solid #f1f5f9;'});
      const tdConc = node('td', null, row, {style: 'padding:5px 8px;border-bottom:1px solid #f1f5f9;'});
      renderBadge(item.conclusion, tdConc);
      const tdCount = node('td', null, row, {style: 'padding:5px 8px;border-bottom:1px solid #f1f5f9;text-align:right;'});
      node('span', `${item.count} lượt`, tdCount, {style: 'font-weight:700;color:#1e293b;font-size:11px;'});
    }
  }
}
