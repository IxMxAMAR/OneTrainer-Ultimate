// OneTrainer WebUI Application State & Realtime Sync

const ARCHITECTURE_METADATA = {
  KREA_2: {
    category: "Krea 2",
    base_model: "krea/Krea-2-Raw",
    resolution: "512",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: false,
    sample_prompt: "masterpiece, highly detailed 8k portrait of a cyberpunk character in neon city rain\ncinematic wide shot of a futuristic landscape at golden hour, sharp focus"
  },
  FLUX_DEV_1: {
    category: "Flux Dev.1",
    base_model: "black-forest-labs/FLUX.1-dev",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: true,
    sample_prompt: "photo of a serene woman walking in an autumn forest, golden hour lighting, 8k uhd, dslr, natural skin texture\na vibrant street market at night with neon lights and reflections on wet asphalt"
  },
  FLUX_1_SCHNELL: {
    category: "Flux Dev.1",
    base_model: "black-forest-labs/FLUX.1-schnell",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: true,
    sample_prompt: "a majestic eagle flying over snow-covered mountains, highly detailed, dramatic lighting"
  },
  FLUX_2: {
    category: "Flux 2",
    base_model: "black-forest-labs/FLUX.2-dev",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: true,
    sample_prompt: "ultra detailed cinematic portrait, 8k resolution, subsurface scattering"
  },
  STABLE_DIFFUSION_XL_10_BASE: {
    category: "SDXL",
    base_model: "stabilityai/stable-diffusion-xl-base-1.0",
    resolution: "1024",
    format: "KOHYA_LORA",
    is_dit: false,
    has_te2: true,
    sample_prompt: "masterpiece, best quality, portrait of a warrior with ornate armor, cinematic lighting, photorealistic\na cozy cabin in the woods during a winter snowfall, warm light from windows"
  },
  STABLE_DIFFUSION_15: {
    category: "SD1.5",
    base_model: "runwayml/stable-diffusion-v1-5",
    resolution: "512",
    format: "KOHYA_LORA",
    is_dit: false,
    has_te2: false,
    sample_prompt: "masterpiece, best quality, 1girl, solo, looking at viewer, detailed eyes, cinematic lighting\na fantasy castle surrounded by glowing floating crystals, digital painting"
  },
  STABLE_DIFFUSION_3: {
    category: "SD3",
    base_model: "stabilityai/stable-diffusion-3-medium-diffusers",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: true,
    sample_prompt: "a close up portrait of a futuristic astronaut with reflections in the visor, high detail, 8k"
  },
  STABLE_DIFFUSION_35: {
    category: "SD3",
    base_model: "stabilityai/stable-diffusion-3.5-large",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: true,
    sample_prompt: "a close up portrait of a futuristic astronaut with reflections in the visor, high detail, 8k"
  },
  SANA: {
    category: "Sana",
    base_model: "Efficient-Large-Model/Sana_1600M_1024px",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: false,
    sample_prompt: "artistic digital illustration of an enchanted forest with bioluminescent mushrooms, vibrant colors"
  },
  HUNYUAN_VIDEO: {
    category: "Hunyuan Video",
    base_model: "tencent/HunyuanVideo",
    resolution: "720",
    format: "COMFY_LORA",
    is_dit: true,
    has_te2: true,
    sample_prompt: "a cinematic drone shot moving through a mountain valley with waterfalls, golden sunlight"
  },
  QWEN: {
    category: "QwenImage",
    base_model: "Qwen/Qwen2-VL-7B-Instruct",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: false,
    sample_prompt: "a high quality portrait of a student in a library reading a book, soft lighting"
  },
  ANIMA: {
    category: "Anima",
    base_model: "cirno-ai/Anima-7B",
    resolution: "1024",
    format: "KOHYA_LORA",
    is_dit: true,
    has_te2: false,
    sample_prompt: "masterpiece anime style illustration of a magical girl with glowing staff"
  },
  Z_IMAGE: {
    category: "Z-Image",
    base_model: "Z-Image-Ai/Z-Image",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: false,
    sample_prompt: "stunning high-res landscape photograph with mountain range and clear reflection"
  },
  CHROMA_1: {
    category: "Chroma",
    base_model: "lodestones/Chroma-Radiant",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: false,
    sample_prompt: "a vibrant sci-fi illustration with volumetric lighting and intense color grading"
  },
  ERNIE: {
    category: "Ernie Image",
    base_model: "baidu/Ernie-Image",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: false,
    sample_prompt: "a detailed digital illustration of traditional Chinese architecture surrounded by pine trees"
  },
  IDEOGRAM_4: {
    category: "Ideogram 4",
    base_model: "ideogram-ai/ideogram-v4",
    resolution: "1024",
    format: "COMFY_LORA",
    is_dit: true,
    has_te2: false,
    sample_prompt: "a vintage typography poster with intricate letterforms and distressed paper texture"
  }
};

let currentConfig = {};
let allPresets = {};
let isTraining = false;
let lossChart = null;
let ws = null;
let currentBrowseTargetInput = null;

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

// Update UI visibility and render quick preset buttons for active architecture
function updateArchitectureUI(archType) {
  const meta = ARCHITECTURE_METADATA[archType] || {};
  const cat = meta.category || archType;

  document.getElementById('activeArchLabel').textContent = `${archType} Presets:`;

  // Render authentic OneTrainer presets for this architecture
  const pillsContainer = document.getElementById('archPresetPills');
  pillsContainer.innerHTML = '';

  const presetsForCat = allPresets[cat] || [];
  if (presetsForCat.length > 0) {
    presetsForCat.forEach(p => {
      const btn = document.createElement('button');
      btn.className = 'preset-pill-btn';
      btn.textContent = `⚡ ${p.name}`;
      btn.onclick = () => loadPresetFile(p.category, p.filename);
      pillsContainer.appendChild(btn);
    });
  } else {
    pillsContainer.innerHTML = '<span style="font-size: 12px; color: #9ca3af;">Custom architecture configuration</span>';
  }

  // Dynamic Visibility: DiT (Transformer) vs UNet fields
  const transformerField = document.getElementById('transformerDtypeGroup');
  const unetField = document.getElementById('unetDtypeGroup');
  const offloadField = document.getElementById('transformerOffloadGroup');
  const te2Field = document.getElementById('te2DtypeGroup');

  if (meta.is_dit) {
    if (transformerField) transformerField.style.display = 'flex';
    if (offloadField) offloadField.style.display = 'flex';
    if (unetField) unetField.style.display = 'none';
  } else {
    if (transformerField) transformerField.style.display = 'none';
    if (offloadField) offloadField.style.display = 'none';
    if (unetField) unetField.style.display = 'flex';
  }

  if (te2Field) {
    te2Field.style.display = meta.has_te2 ? 'flex' : 'none';
  }
}

// Listen to Model Architecture dropdown changes
document.getElementById('model_type').addEventListener('change', (e) => {
  const chosenArch = e.target.value;
  const meta = ARCHITECTURE_METADATA[chosenArch];
  if (meta) {
    const presetsForCat = allPresets[meta.category] || [];
    if (presetsForCat.length > 0) {
      loadPresetFile(presetsForCat[0].category, presetsForCat[0].filename);
    } else {
      document.getElementById('base_model_name').value = meta.base_model || '';
      document.getElementById('output_model_format').value = meta.format || 'DIFFUSERS_LORA';
      if (meta.sample_prompt) document.getElementById('sample_prompts').value = meta.sample_prompt;
      updateArchitectureUI(chosenArch);
    }
  } else {
    updateArchitectureUI(chosenArch);
  }
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
    if (!currentConfig.base_model_name && currentConfig.model_type) {
      const meta = ARCHITECTURE_METADATA[currentConfig.model_type] || {};
      currentConfig.base_model_name = meta.base_model || '';
    }
    populateForm(currentConfig);
    document.getElementById('base_model_name').value = currentConfig.base_model_name || '';
    if (currentConfig.model_type) {
      updateArchitectureUI(currentConfig.model_type);
    }
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
    if (val !== undefined && val !== null) {
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
    container.innerHTML = '<div class="empty-placeholder">No concepts configured. Click "1-Click Auto-Detect" or "+ Add Concept Manually" above.</div>';
    return;
  }

  concepts.forEach((c, idx) => {
    const card = document.createElement('div');
    card.className = 'concept-card';
    const cPath = c.path || c.image_folder || '';
    card.innerHTML = `
      <div class="concept-card-top">
        <div style="display: flex; align-items: center; gap: 8px;">
          <strong style="color: #fff;">${c.name || `Concept #${idx + 1}`}</strong>
          ${c.image_count ? `<span class="concept-badge">${c.image_count} Images (${c.caption_count || 0} Captions)</span>` : ''}
        </div>
        <button class="btn-remove-concept" onclick="removeConcept(${idx})">&times;</button>
      </div>
      <div class="form-group">
        <label>Image Directory</label>
        <div class="input-with-btn">
          <input type="text" id="concept_img_${idx}" value="${cPath}" onchange="updateConcept(${idx}, 'path', this.value)" placeholder="/workspace/datasets/my_concept" />
          <button class="btn btn-secondary btn-browse" onclick="openFileBrowser('concept_img_${idx}')">📁</button>
        </div>
      </div>
      <div class="form-row">
        <div class="form-group flex-1">
          <label>Repeats</label>
          <input type="number" value="${c.balancing || c.repeat || 1}" onchange="updateConcept(${idx}, 'balancing', Number(this.value))" />
        </div>
        <div class="form-group flex-1">
          <label>Resolution</label>
          <input type="text" value="${c.resolution || '512'}" onchange="updateConcept(${idx}, 'resolution', this.value)" />
        </div>
      </div>
      <div class="form-row">
        <div class="form-group flex-1">
          <label>Mask Folder (Optional)</label>
          <div class="input-with-btn">
            <input type="text" id="concept_mask_${idx}" value="${c.mask_folder || ''}" onchange="updateConcept(${idx}, 'mask_folder', this.value)" placeholder="Optional mask path" />
            <button class="btn btn-secondary btn-browse" onclick="openFileBrowser('concept_mask_${idx}')">📁</button>
          </div>
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
    if (field === 'path') currentConfig.concepts[idx]['image_folder'] = val;
  }
};

document.getElementById('btnAddConcept').addEventListener('click', () => {
  if (!currentConfig.concepts) currentConfig.concepts = [];
  currentConfig.concepts.push({
    name: `Concept_${currentConfig.concepts.length + 1}`,
    path: '/workspace/datasets',
    image_folder: '/workspace/datasets',
    balancing: 1.0,
    repeat: 1,
    resolution: '512',
    caption_extension: '.txt',
    mask_folder: ''
  });
  renderConcepts(currentConfig.concepts);
});

// 1-Click Dataset Auto-Detector
document.getElementById('btnAutoDetectDatasets').addEventListener('click', async () => {
  try {
    const res = await fetch('/api/datasets/auto_detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });
    const data = await res.json();
    if (data.concepts && data.concepts.length > 0) {
      currentConfig.concepts = data.concepts;
      renderConcepts(currentConfig.concepts);
      alert(`Auto-detected ${data.concepts.length} concept dataset folders!`);
    } else {
      alert('No image folders found in /workspace/datasets. Please place your image folders there or browse manually.');
    }
  } catch (e) {
    alert(`Error scanning datasets: ${e}`);
  }
});

// Load Preset File directly from backend
window.loadPresetFile = async function(cat, file) {
  try {
    const res = await fetch(`/api/preset/${encodeURIComponent(cat)}/${encodeURIComponent(file)}`);
    const data = await res.json();
    if (data.status === 'ok') {
      currentConfig = data.config;

      const modelType = currentConfig.model_type;
      const meta = ARCHITECTURE_METADATA[modelType] || {};

      if (!currentConfig.base_model_name && meta.base_model) {
        currentConfig.base_model_name = meta.base_model;
      }

      // Populate forms
      populateForm(currentConfig);

      // Explicitly set base model input value
      document.getElementById('base_model_name').value = currentConfig.base_model_name || meta.base_model || '';

      // Auto-name output destination if generic
      if (!currentConfig.output_model_destination || currentConfig.output_model_destination.startsWith('models/')) {
        const safeCat = cat.toLowerCase().replace(/\s+/g, '_');
        const safeName = file.replace(/#/g, '').replace(/\.json/g, '').toLowerCase().replace(/\s+/g, '_');
        const targetDest = `output/${safeCat}_${safeName}.safetensors`;
        document.getElementById('output_model_destination').value = targetDest;
        setNestedValue(currentConfig, 'output_model_destination', targetDest);
      }

      // Sample Prompts
      if (meta.sample_prompt) {
        document.getElementById('sample_prompts').value = meta.sample_prompt;
      }

      updateArchitectureUI(modelType);
      renderConcepts(currentConfig.concepts || []);

      // Highlight active pill
      document.querySelectorAll('.preset-pill-btn').forEach(btn => {
        btn.classList.toggle('active', btn.textContent.includes(file.replace(/#/g, '').replace(/\.json/g, '')));
      });

      alert(`✅ Loaded authentic OneTrainer preset: ${cat} (${file})\nBase Model: ${currentConfig.base_model_name}\nTraining Method: ${currentConfig.training_method}\nLearning Rate: ${currentConfig.learning_rate}`);
    }
  } catch (err) {
    alert(`Failed to load preset: ${err}`);
  }
};

// Preset Dropdown Loader
async function fetchPresets() {
  try {
    const res = await fetch('/api/presets');
    allPresets = await res.json();
    const select = document.getElementById('presetSelect');
    select.innerHTML = '<option value="" disabled selected>⚡ All Architecture Presets (60+)...</option>';
    for (const [cat, list] of Object.entries(allPresets)) {
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

document.getElementById('presetSelect').addEventListener('change', (e) => {
  const val = e.target.value;
  if (!val) return;
  const [cat, file] = val.split('/');
  loadPresetFile(cat, file);
});

// 1-Click Export / Download Config JSON to Computer
document.getElementById('btnExportConfig').addEventListener('click', () => {
  syncFormToConfig();
  const configJson = JSON.stringify(currentConfig, null, 2);

  // Trigger browser download
  const blob = new Blob([configJson], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const archName = (currentConfig.model_type || 'training').toLowerCase();
  a.href = url;
  a.download = `${archName}_config.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  // Also display modal with Copy button
  const modalOverlay = document.getElementById('modalOverlay');
  const modalTitle = document.getElementById('modalTitle');
  const modalBody = document.getElementById('modalBody');

  modalTitle.textContent = '📥 Training Config JSON';
  modalBody.innerHTML = `
    <div style="margin-bottom: 10px; font-size: 13px; color: #10b981;">
      ✅ Config downloaded to your computer as <code>${archName}_config.json</code>!
    </div>
    <div class="form-group">
      <textarea id="exportJsonText" rows="14" style="font-family: monospace; font-size: 11px; background: #0b0d14;" readonly>${configJson}</textarea>
    </div>
    <button class="btn btn-primary" onclick="copyExportedJson()">📋 Copy to Clipboard</button>
  `;
  modalOverlay.classList.remove('hidden');
});

window.copyExportedJson = function() {
  const textarea = document.getElementById('exportJsonText');
  textarea.select();
  navigator.clipboard.writeText(textarea.value);
  alert('Copied training config JSON to clipboard!');
};

// Interactive File / Folder Browser Modal
window.openFileBrowser = async function(targetInputId, startPath = null) {
  currentBrowseTargetInput = targetInputId;
  const modalOverlay = document.getElementById('modalOverlay');
  const modalTitle = document.getElementById('modalTitle');
  const modalBody = document.getElementById('modalBody');

  modalTitle.textContent = '📁 Select Folder or File';
  modalBody.innerHTML = '<div style="padding: 20px; text-align: center;">Loading directory...</div>';
  modalOverlay.classList.remove('hidden');

  try {
    const url = startPath ? `/api/browse?path=${encodeURIComponent(startPath)}` : '/api/browse';
    const res = await fetch(url);
    const data = await res.json();

    let html = `
      <div style="margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; font-size: 13px;">
        <span style="color: #a855f7; font-family: monospace; word-break: break-all;">${data.current_path}</span>
        <button class="btn btn-primary" onclick="selectPath('${data.current_path.replace(/\\/g, '/')}')">Select Current Folder</button>
      </div>
      <div class="browser-list">
    `;

    if (data.parent_path) {
      html += `
        <div class="browser-item" onclick="openFileBrowser('${targetInputId}', '${data.parent_path.replace(/\\/g, '/')}')">
          <span>📁 .. (Up to parent)</span>
        </div>
      `;
    }

    data.items.forEach(item => {
      const cleanPath = item.path.replace(/\\/g, '/');
      if (item.is_dir) {
        html += `
          <div class="browser-item" onclick="openFileBrowser('${targetInputId}', '${cleanPath}')">
            <span>📁 ${item.name}</span>
            <button class="btn btn-secondary" onclick="event.stopPropagation(); selectPath('${cleanPath}')">Select</button>
          </div>
        `;
      } else {
        html += `
          <div class="browser-item" onclick="selectPath('${cleanPath}')">
            <span>📄 ${item.name}</span>
            <button class="btn btn-secondary">Select</button>
          </div>
        `;
      }
    });

    html += '</div>';
    modalBody.innerHTML = html;
  } catch (e) {
    modalBody.innerHTML = `<div style="color: #ef4444; padding: 20px;">Error browsing: ${e}</div>`;
  }
};

window.selectPath = function(chosenPath) {
  if (currentBrowseTargetInput) {
    const input = document.getElementById(currentBrowseTargetInput);
    if (input) {
      input.value = chosenPath;
      input.dispatchEvent(new Event('change'));
    }
  }
  document.getElementById('modalOverlay').classList.add('hidden');
};

// Model Hub & Downloader Modal
document.getElementById('btnModelHub').addEventListener('click', async () => {
  const modalOverlay = document.getElementById('modalOverlay');
  const modalTitle = document.getElementById('modalTitle');
  const modalBody = document.getElementById('modalBody');

  modalTitle.textContent = '📥 Model Hub — 1-Click Hugging Face Downloader';
  modalOverlay.classList.remove('hidden');

  try {
    const res = await fetch('/api/models/presets');
    const models = await res.json();

    let html = '<div class="model-hub-grid">';
    models.forEach(m => {
      html += `
        <div class="model-hub-card">
          <div class="model-hub-title">${m.name}</div>
          <div style="font-size: 11px; color: #9ca3af; font-family: monospace;">${m.repo} (${m.size})</div>
          ${m.gated ? '<div style="font-size: 11px; color: #f59e0b;">Requires HF Token in Model tab</div>' : ''}
          <div style="display: flex; gap: 6px; margin-top: 6px;">
            <button class="btn btn-primary" onclick="downloadModelHub('${m.repo}')">⚡ Download to /workspace</button>
            <button class="btn btn-secondary" onclick="useModelInConfig('${m.repo}', '${m.type}')">Use as Base</button>
          </div>
        </div>
      `;
    });
    html += '</div>';
    modalBody.innerHTML = html;
  } catch (e) {
    modalBody.innerHTML = `<div>Error loading model hub: ${e}</div>`;
  }
});

window.downloadModelHub = async function(repoId) {
  try {
    const res = await fetch('/api/models/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo_id: repoId })
    });
    const data = await res.json();
    alert(`Started background download for ${repoId} into ${data.dest}! You can watch the download status in the Live Dashboard logs.`);
  } catch (e) {
    alert(`Failed to trigger download: ${e}`);
  }
};

window.useModelInConfig = function(repoId, modelType) {
  document.getElementById('base_model_name').value = repoId;
  document.getElementById('model_type').value = modelType;
  updateArchitectureUI(modelType);
  syncFormToConfig();
  document.getElementById('modalOverlay').classList.add('hidden');
  alert(`Base model set to ${repoId} (${modelType})!`);
};

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
      if (currentConfig.model_type) {
        updateArchitectureUI(currentConfig.model_type);
      }
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
  fetchPresets().then(() => {
    fetchConfig();
  });
});
