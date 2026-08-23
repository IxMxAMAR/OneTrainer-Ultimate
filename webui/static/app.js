// OneTrainer WebUI Application State & Realtime Sync

let currentConfig = {};
let isTraining = false;
let lossChart = null;
let ws = null;

// Initialize Chart.js
function initLossChart() {
  const ctx = document.getElementById('lossChart').getContext('2d');
  lossChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Training Loss',
        data: [],
        borderColor: '#6366f1',
        backgroundColor: 'rgba(99, 102, 241, 0.1)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.2,
        fill: true
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: {
          grid: { color: '#1f2937' },
          ticks: { color: '#9ca3af', maxTicksLimit: 10 }
        },
        y: {
          grid: { color: '#1f2937' },
          ticks: { color: '#9ca3af' }
        }
      },
      plugins: {
        legend: { labels: { color: '#f3f4f6' } }
      }
    }
  });
}

// Tab Switching
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const tabId = btn.getAttribute('data-tab');
    document.getElementById(tabId).classList.add('active');
  });
});

// WebSocket Connection
function connectWebSocket() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${location.host}/ws`);

  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === 'init') {
      updateStatus(message.status);
      if (message.loss_history) renderLossHistory(message.loss_history);
      if (message.logs) renderLogs(message.logs);
    } else if (message.type === 'status') {
      updateStatus(message.data);
    } else if (message.type === 'sample') {
      appendSampleImage(message.data);
    }
  };

  ws.onclose = () => {
    setTimeout(connectWebSocket, 2000);
  };
}

function updateStatus(status) {
  isTraining = status.is_training;
  const statusPill = document.getElementById('statusPill');
  const statusText = document.getElementById('statusText');
  const etaPill = document.getElementById('etaPill');
  const btnStart = document.getElementById('btnStartTraining');
  const btnSample = document.getElementById('btnSampleNow');
  const btnSave = document.getElementById('btnSaveCheckpoint');

  statusText.textContent = status.status || 'Idle';
  etaPill.textContent = `ETA: ${status.eta || '--:--'}`;

  if (isTraining) {
    statusPill.className = 'status-indicator training';
    btnStart.textContent = '⏹ Stop Training';
    btnStart.className = 'btn btn-primary danger';
    btnSample.disabled = false;
    btnSave.disabled = false;
  } else {
    statusPill.className = status.status && status.status.startsWith('Error') ? 'status-indicator error' : 'status-indicator';
    btnStart.textContent = '▶ Start Training';
    btnStart.className = 'btn btn-primary';
    btnSample.disabled = true;
    btnSave.disabled = true;
  }

  // Progress Bars
  const step = status.step || 0;
  const maxStep = status.max_step || 100;
  const epoch = status.epoch || 0;
  const maxEpoch = status.max_epoch || 1;

  document.getElementById('stepCounter').textContent = `${step} / ${maxStep}`;
  document.getElementById('epochCounter').textContent = `${epoch} / ${maxEpoch}`;

  const stepPct = maxStep > 0 ? Math.min(100, Math.round((step / maxStep) * 100)) : 0;
  const epochPct = maxEpoch > 0 ? Math.min(100, Math.round((epoch / maxEpoch) * 100)) : 0;

  document.getElementById('stepProgressBar').style.width = `${stepPct}%`;
  document.getElementById('epochProgressBar').style.width = `${epochPct}%`;

  // GPU Resource Metrics
  if (status.gpu && status.gpu.name) {
    document.getElementById('gpuName').textContent = status.gpu.name;
    document.getElementById('gpuAllocated').textContent = `${status.gpu.allocated_gb} GB`;
    document.getElementById('gpuReserved').textContent = `${status.gpu.reserved_gb} GB`;
    document.getElementById('gpuTotal').textContent = `${status.gpu.total_gb} GB`;
  }
}

function renderLossHistory(history) {
  if (!lossChart) return;
  lossChart.data.labels = history.map(h => h.step);
  lossChart.data.datasets[0].data = history.map(h => h.loss);
  lossChart.update();
}

function renderLogs(logs) {
  const terminal = document.getElementById('logTerminal');
  terminal.innerHTML = logs.map(l => `<div>${l}</div>`).join('');
  terminal.scrollTop = terminal.scrollHeight;
}

function appendSampleImage(sample) {
  const gallery = document.getElementById('sampleGallery');
  const empty = gallery.querySelector('.empty-placeholder');
  if (empty) empty.remove();

  const card = document.createElement('div');
  card.className = 'sample-img-card';
  card.innerHTML = `
    <img src="${sample.image}" alt="Sample step ${sample.step}" />
    <div style="font-size: 11px; color: #9ca3af; text-align: center; margin-top: 4px;">
      Step ${sample.step} (Epoch ${sample.epoch})
    </div>
  `;
  gallery.prepend(card);
}

// Config Binding
async function fetchConfig() {
  try {
    const res = await fetch('/api/config');
    currentConfig = await res.json();
    populateForm(currentConfig);
    renderConcepts(currentConfig.concepts || []);
  } catch (e) {
    console.error('Error fetching config:', e);
  }
}

function getNestedValue(obj, path) {
  return path.split('.').reduce((acc, part) => acc && acc[part] !== undefined ? acc[part] : undefined, obj);
}

function setNestedValue(obj, path, val) {
  const parts = path.split('.');
  let current = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    if (!current[parts[i]]) current[parts[i]] = {};
    current = current[parts[i]];
  }
  current[parts[parts.length - 1]] = val;
}

function populateForm(config) {
  document.querySelectorAll('[data-bind]').forEach(input => {
    const path = input.getAttribute('data-bind');
    const val = getNestedValue(config, path);
    if (val !== undefined) {
      if (input.type === 'checkbox') {
        input.checked = Boolean(val);
      } else {
        input.value = val;
      }
    }
  });
}

function syncFormToConfig() {
  document.querySelectorAll('[data-bind]').forEach(input => {
    const path = input.getAttribute('data-bind');
    let val;
    if (input.type === 'checkbox') {
      val = input.checked;
    } else if (input.type === 'number') {
      val = input.value === '' ? 0 : Number(input.value);
    } else {
      val = input.value;
    }
    setNestedValue(currentConfig, path, val);
  });
  return currentConfig;
}

// Concept Manager
function renderConcepts(concepts) {
  const container = document.getElementById('conceptsList');
  container.innerHTML = '';
  if (!concepts || concepts.length === 0) {
    container.innerHTML = '<div class="empty-placeholder">No concepts configured. Click "+ Add New Concept" above.</div>';
    return;
  }

  concepts.forEach((c, idx) => {
    const card = document.createElement('div');
    card.className = 'concept-card';
    card.innerHTML = `
      <div class="concept-card-top">
        <strong style="color: #fff;">Concept #${idx + 1}: ${c.name || 'Default Concept'}</strong>
        <button class="btn-remove-concept" onclick="removeConcept(${idx})">&times;</button>
      </div>
      <div class="form-group">
        <label>Image Directory</label>
        <input type="text" value="${c.image_folder || ''}" onchange="updateConcept(${idx}, 'image_folder', this.value)" placeholder="/workspace/datasets/my_concept" />
      </div>
      <div class="form-row">
        <div class="form-group flex-1">
          <label>Repeats</label>
          <input type="number" value="${c.repeat || 1}" onchange="updateConcept(${idx}, 'repeat', Number(this.value))" />
        </div>
        <div class="form-group flex-1">
          <label>Resolution</label>
          <input type="text" value="${c.resolution || '1024'}" onchange="updateConcept(${idx}, 'resolution', this.value)" />
        </div>
      </div>
      <div class="form-row">
        <div class="form-group flex-1">
          <label>Mask Folder (Optional)</label>
          <input type="text" value="${c.mask_folder || ''}" onchange="updateConcept(${idx}, 'mask_folder', this.value)" placeholder="Optional mask path" />
        </div>
        <div class="form-group flex-1">
          <label>Caption Extension</label>
          <input type="text" value="${c.caption_extension || '.txt'}" onchange="updateConcept(${idx}, 'caption_extension', this.value)" />
        </div>
      </div>
    `;
    container.appendChild(card);
  });
}

window.removeConcept = function(idx) {
  if (currentConfig.concepts) {
    currentConfig.concepts.splice(idx, 1);
    renderConcepts(currentConfig.concepts);
  }
};

window.updateConcept = function(idx, field, val) {
  if (currentConfig.concepts && currentConfig.concepts[idx]) {
    currentConfig.concepts[idx][field] = val;
  }
};

document.getElementById('btnAddConcept').addEventListener('click', () => {
  if (!currentConfig.concepts) currentConfig.concepts = [];
  currentConfig.concepts.push({
    name: `Concept_${currentConfig.concepts.length + 1}`,
    image_folder: '/workspace/datasets',
    repeat: 1,
    resolution: '1024',
    caption_extension: '.txt',
    mask_folder: ''
  });
  renderConcepts(currentConfig.concepts);
});

// Preset Dropdown Loader
async function fetchPresets() {
  try {
    const res = await fetch('/api/presets');
    const presets = await res.json();
    const select = document.getElementById('presetSelect');
    select.innerHTML = '<option value="" disabled selected>⚡ Load Architecture Preset...</option>';
    for (const [cat, list] of Object.entries(presets)) {
      const optGroup = document.createElement('optgroup');
      optGroup.label = cat;
      list.forEach(p => {
        const opt = document.createElement('option');
        opt.value = `${p.category}/${p.filename}`;
        opt.textContent = `${cat} — ${p.name}`;
        optGroup.appendChild(opt);
      });
      select.appendChild(optGroup);
    }
  } catch (e) {
    console.error('Error loading presets:', e);
  }
}

document.getElementById('presetSelect').addEventListener('change', async (e) => {
  const val = e.target.value;
  if (!val) return;
  const [cat, file] = val.split('/');
  try {
    const res = await fetch(`/api/preset/${cat}/${file}`);
    const data = await res.json();
    if (data.status === 'ok') {
      currentConfig = data.config;
      populateForm(currentConfig);
      renderConcepts(currentConfig.concepts || []);
      alert(`Loaded preset: ${cat}/${file}`);
    }
  } catch (err) {
    alert(`Failed to load preset: ${err}`);
  }
});

// Start / Stop Training
document.getElementById('btnStartTraining').addEventListener('click', async () => {
  if (isTraining) {
    if (confirm('Stop the current training run?')) {
      await fetch('/api/train/stop', { method: 'POST' });
    }
  } else {
    syncFormToConfig();
    await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(currentConfig)
    });
    const res = await fetch('/api/train/start', { method: 'POST' });
    if (!res.ok) {
      const err = await res.json();
      alert(`Error starting training: ${err.detail || 'Unknown error'}`);
    }
  }
});

document.getElementById('btnSampleNow').addEventListener('click', async () => {
  await fetch('/api/train/sample_now', { method: 'POST' });
});

document.getElementById('btnSaveCheckpoint').addEventListener('click', async () => {
  await fetch('/api/train/save_now', { method: 'POST' });
});

// Modals for Save/Load
const modalOverlay = document.getElementById('modalOverlay');
const modalTitle = document.getElementById('modalTitle');
const modalBody = document.getElementById('modalBody');
document.getElementById('modalClose').addEventListener('click', () => modalOverlay.classList.add('hidden'));

document.getElementById('btnSaveConfig').addEventListener('click', () => {
  modalTitle.textContent = 'Save Training Configuration';
  modalBody.innerHTML = `
    <div class="form-group">
      <label>Config Filename</label>
      <input type="text" id="saveFilenameInput" value="my_training_config.json" />
    </div>
    <button id="btnConfirmSave" class="btn btn-primary mt-2">Save JSON</button>
  `;
  modalOverlay.classList.remove('hidden');
  document.getElementById('btnConfirmSave').addEventListener('click', async () => {
    syncFormToConfig();
    await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(currentConfig)
    });
    const filename = document.getElementById('saveFilenameInput').value;
    const res = await fetch('/api/save_config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename })
    });
    if (res.ok) {
      alert('Config saved successfully!');
      modalOverlay.classList.add('hidden');
    }
  });
});

document.getElementById('btnLoadConfig').addEventListener('click', async () => {
  modalTitle.textContent = 'Load Training Configuration';
  const res = await fetch('/api/saved_configs');
  const configs = await res.json();
  modalBody.innerHTML = `
    <div class="form-group">
      <label>Select Saved Config</label>
      <select id="loadConfigSelect">
        ${configs.map(c => `<option value="${c}">${c}</option>`).join('')}
      </select>
    </div>
    <button id="btnConfirmLoad" class="btn btn-primary mt-2">Load Selected</button>
  `;
  modalOverlay.classList.remove('hidden');
  document.getElementById('btnConfirmLoad').addEventListener('click', async () => {
    const filename = document.getElementById('loadConfigSelect').value;
    const loadRes = await fetch('/api/load_config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename })
    });
    const data = await loadRes.json();
    if (data.status === 'ok') {
      currentConfig = data.config;
      populateForm(currentConfig);
      renderConcepts(currentConfig.concepts || []);
      modalOverlay.classList.add('hidden');
      alert(`Loaded: ${filename}`);
    }
  });
});

// Startup
window.addEventListener('DOMContentLoaded', () => {
  initLossChart();
  connectWebSocket();
  fetchConfig();
  fetchPresets();
});
