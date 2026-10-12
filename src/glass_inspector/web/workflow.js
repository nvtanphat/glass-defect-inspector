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
function clearWorkflow() {
  // Saved images may still be displayed or downloadable in Workspace.
  workflowObjectUrls = workflowObjectUrls.filter(url => {
    if (url === appState.currentImage || url === appState.annotatedImage) return true;
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
  node('h2', 'Lịch sử kiểm tra', workflowScreen);
  const controls = node('div', null, workflowScreen, {'data-card': ''});
  const q = node('input', null, controls, {placeholder: 'Tìm mã lượt kiểm tra', 'aria-label': 'Tìm mã lượt kiểm tra'});
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
    for (const [key, value] of Object.entries({q: q.value, status: status.value, conclusion: conclusion.value, model_id: model.value})) {
      if (value) historyFilters[key] = value;
    }
    historyPage = 1; await loadHistory();
  });
  const table = node('table', null, workflowScreen, {'data-card': ''});
  const header = node('tr', null, node('thead', null, table));
  for (const label of ['Mã lượt kiểm tra', 'Thời gian', 'Bộ kiểm tra', 'Tình trạng', 'Kết luận', 'Thao tác']) node('th', label, header);
  const body = node('tbody', null, table);
  for (const item of data.items) {
    const row = node('tr', null, body);
    for (const value of [item.id, new Date(item.created_at).toLocaleString(), modelLabel(item.model_id), statusLabel(item.status), qualityLabel(item.conclusion)]) node('td', value, row);
    button('Xem và xác nhận', node('td', null, row), () => openInspection(item.id), {'data-inspection-id': item.id});
  }
  if (!data.items.length) node('p', 'Chưa có kết quả phù hợp. Hãy thử điều kiện tìm kiếm khác.', workflowScreen);
  const pager = node('div', null, workflowScreen);
  const previous = button('Trang trước', pager, async () => { historyPage--; await loadHistory(); });
  previous.disabled = historyPage === 1;
  node('span', `Trang ${data.page} · ${data.total} lượt kiểm tra`, pager);
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
  const card = node('div', null, workflowScreen, {'data-card': ''});
  node('h2', `Xác nhận kết quả — ${detail.id}`, card);
  node('p', `${statusLabel(detail.status)} / ${qualityLabel(detail.conclusion)} · ${modelLabel(detail.model_id)}`, card, {id: 'review-summary'});
  node('p', assessmentText(detail), card);
  for (const format of ['json', 'csv', 'pdf']) button(({json: 'Tải dữ liệu', csv: 'Tải bảng kết quả', pdf: 'Tải báo cáo PDF'})[format], card, () => exportInspection(id, format));
  for (const image of detail.images) {
    const imageCard = node('div', null, workflowScreen, {'data-card': ''});
    node('h3', image.filename, imageCard);
    const src = await imageUrl(image.annotated_url || image.original_url);
    if (epoch !== workflowEpoch) return;
    const preview = node('img', null, imageCard, {src, alt: 'Ảnh kết quả đã lưu'});
    preview.style.maxHeight = '400px'; preview.style.maxWidth = '100%';
    const result = image.result;
    node('p', result ? `${result.total_defects} lỗi · ${durationLabel(result.latency_ms)}` : 'Chưa có kết quả hoàn chỉnh', imageCard);
    if (result) {
      const table = node('table', null, imageCard);
      for (const d of result.detections) {
        const row = node('tr', null, table);
        for (const value of [defectLabel(d.class_name), `${(d.confidence * 100).toFixed(0)}%`]) node('td', value, row);
      }
      button('Xem vị trí lỗi trên ảnh', imageCard, async () => {
        const epoch = workflowEpoch;
        const original = await imageUrl(image.original_url);
        if (epoch !== workflowEpoch) return;
        displayImage(original);
        // Use original coordinate space. Saved annotated image stays downloadable.
        appState.imageWidth = result.image_width; appState.imageHeight = result.image_height;
        handleInferenceResult({...result, success: true, inspection_id: id, conclusion: detail.conclusion, annotated_b64: src});
        await showScreen('workspace');
      });
    }
  }
  if (detail.status === 'COMPLETED') {
    const form = node('form', null, workflowScreen, {'data-card': '', id: 'review-form'});
    node('h3', 'Xác nhận của người kiểm tra', form);
    const choice = node('select', null, form, {id: 'review-conclusion', 'aria-label': 'Kết luận xác nhận'});
    for (const value of ['PASS', 'REJECT', 'REVIEW']) choice.add(new Option(qualityLabel(value), value));
    choice.value = detail.conclusion;
    const reviewer = node('input', null, form, {id: 'reviewer', required: '', maxlength: '120', placeholder: 'Tên người kiểm tra', 'aria-label': 'Tên người kiểm tra'});
    const reason = node('textarea', null, form, {id: 'review-reason', required: '', minlength: '3', maxlength: '2000', placeholder: 'Lý do xác nhận hoặc thay đổi kết luận', 'aria-label': 'Lý do xác nhận'});
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
  const history = node('div', null, workflowScreen, {'data-card': ''});
  node('h3', 'Lịch sử xác nhận', history);
  if (!detail.reviews.length) node('p', 'Chưa có người xác nhận kết quả này.', history);
  for (const review of detail.reviews) node('p', `v${review.version} · ${review.reviewer} · ${qualityLabel(review.previous_conclusion)} → ${qualityLabel(review.conclusion)} · ${review.reason} · ${new Date(review.created_at).toLocaleString()}`, history);
}

async function loadDashboard(epoch = ++workflowEpoch) {
  const data = await fetchFromApi('/api/dashboard');
  if (epoch !== workflowEpoch) return;
  clearWorkflow(); node('h2', 'Thống kê chất lượng', workflowScreen);
  const metrics = node('div', null, workflowScreen, {'data-card': ''});
  node('p', `Tổng lượt kiểm tra: ${data.total_inspections} · Ảnh đã kiểm tra: ${data.completed_images}`, metrics, {id: 'dashboard-total'});
  node('p', `Thời gian kiểm tra trung bình: ${durationLabel(data.mean_latency_ms)}`, metrics);
  for (const [label, count] of Object.entries(data.statuses)) node('p', `${qualityLabel(defectLabel(statusLabel(label)))}: ${count}`, metrics);
  for (const [title, values] of [['Kết luận chất lượng', data.conclusions], ['Các loại lỗi', data.defect_classes]]) {
    const card = node('div', null, workflowScreen, {'data-card': ''}); node('h3', title, card);
    const max = Math.max(...Object.values(values), 1);
    for (const [label, count] of Object.entries(values)) {
      node('p', `${qualityLabel(defectLabel(statusLabel(label)))}: ${count}`, card);
      const bar = node('div', null, card, {role: 'img', 'aria-label': `${qualityLabel(defectLabel(statusLabel(label)))}: ${count}`});
      bar.style.cssText = `height:12px;background:#2563eb;width:${count / max * 100}%`;
    }
  }
  const trend = node('div', null, workflowScreen, {'data-card': ''}); node('h3', 'Số lượt kiểm tra theo ngày', trend);
  if (!data.trend.length) node('p', 'Chưa có lượt kiểm tra hoàn thành.', trend);
  const table = node('table', null, trend);
  const header = node('tr', null, table); for (const label of ['Ngày', 'Kết luận', 'Số lượt kiểm tra']) node('th', label, header);
  for (const item of data.trend) {
    const row = node('tr', null, table); for (const value of [item.date, qualityLabel(item.conclusion), item.count]) node('td', value, row);
  }
}
