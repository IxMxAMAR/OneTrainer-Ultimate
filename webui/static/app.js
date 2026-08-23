// Architecture metadata map with official base repos, resolutions, formats, and DiT vs UNet flags
const ARCHITECTURE_DEFAULTS = {
  KREA_2: {
    base_model: "krea/Krea-2-Raw",
    resolution: "512",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: false,
    transformer_dtype: "INT_W8A8",
    te_dtype: "FLOAT_8",
    timestep_distribution: "LOGIT_NORMAL",
    sample_prompt: "masterpiece, highly detailed 8k portrait of a cyberpunk character in neon city rain\ncinematic wide shot of a futuristic landscape at golden hour, sharp focus"
  },
  FLUX_DEV_1: {
    base_model: "black-forest-labs/FLUX.1-dev",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: true,
    transformer_dtype: "INT_W8A8",
    te_dtype: "FLOAT_16",
    te2_dtype: "FLOAT_8",
    timestep_distribution: "LOGIT_NORMAL",
    sample_prompt: "photo of a serene woman walking in an autumn forest, golden hour lighting, 8k uhd, dslr, natural skin texture\na vibrant street market at night with neon lights and reflections on wet asphalt"
  },
  FLUX_1_SCHNELL: {
    base_model: "black-forest-labs/FLUX.1-schnell",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: true,
    transformer_dtype: "INT_W8A8",
    te_dtype: "FLOAT_16",
    te2_dtype: "FLOAT_8",
    timestep_distribution: "LOGIT_NORMAL",
    sample_prompt: "a majestic eagle flying over snow-covered mountains, highly detailed, dramatic lighting"
  },
  FLUX_FILL_DEV_1: {
    base_model: "black-forest-labs/FLUX.1-Fill-dev",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: true,
    transformer_dtype: "INT_W8A8",
    te_dtype: "FLOAT_16",
    te2_dtype: "FLOAT_8",
    timestep_distribution: "LOGIT_NORMAL",
    sample_prompt: "a high quality inpainting sample, seamless blend, photographic"
  },
  FLUX_2: {
    base_model: "black-forest-labs/FLUX.2-dev",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: true,
    transformer_dtype: "INT_W8A8",
    te_dtype: "FLOAT_16",
    te2_dtype: "FLOAT_8",
    timestep_distribution: "LOGIT_NORMAL",
    sample_prompt: "ultra detailed cinematic portrait, 8k resolution, subsurface scattering"
  },
  STABLE_DIFFUSION_XL_10_BASE: {
    base_model: "stabilityai/stable-diffusion-xl-base-1.0",
    resolution: "1024",
    format: "KOHYA_LORA",
    is_dit: false,
    has_te2: true,
    unet_dtype: "FLOAT_16",
    te_dtype: "FLOAT_16",
    te2_dtype: "FLOAT_16",
    timestep_distribution: "UNIFORM",
    sample_prompt: "masterpiece, best quality, portrait of a warrior with ornate armor, cinematic lighting, photorealistic\na cozy cabin in the woods during a winter snowfall, warm light from windows"
  },
  STABLE_DIFFUSION_XL_10_BASE_INPAINTING: {
    base_model: "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    resolution: "1024",
    format: "KOHYA_LORA",
    is_dit: false,
    has_te2: true,
    unet_dtype: "FLOAT_16",
    te_dtype: "FLOAT_16",
    te2_dtype: "FLOAT_16",
    timestep_distribution: "UNIFORM",
    sample_prompt: "masterpiece, seamless inpaint portrait, 8k"
  },
  STABLE_DIFFUSION_15: {
    base_model: "runwayml/stable-diffusion-v1-5",
    resolution: "512",
    format: "KOHYA_LORA",
    is_dit: false,
    has_te2: false,
    unet_dtype: "FLOAT_16",
    te_dtype: "FLOAT_16",
    timestep_distribution: "UNIFORM",
    sample_prompt: "masterpiece, best quality, 1girl, solo, looking at viewer, detailed eyes, cinematic lighting\na fantasy castle surrounded by glowing floating crystals, digital painting"
  },
  STABLE_DIFFUSION_15_INPAINTING: {
    base_model: "runwayml/stable-diffusion-inpainting",
    resolution: "512",
    format: "KOHYA_LORA",
    is_dit: false,
    has_te2: false,
    unet_dtype: "FLOAT_16",
    te_dtype: "FLOAT_16",
    timestep_distribution: "UNIFORM",
    sample_prompt: "masterpiece, 1girl inpainting, highly detailed"
  },
  STABLE_DIFFUSION_21: {
    base_model: "stabilityai/stable-diffusion-2-1",
    resolution: "768",
    format: "KOHYA_LORA",
    is_dit: false,
    has_te2: false,
    unet_dtype: "FLOAT_16",
    te_dtype: "FLOAT_16",
    timestep_distribution: "UNIFORM",
    sample_prompt: "a photograph of an astronaut riding a horse on mars, photorealistic"
  },
  STABLE_DIFFUSION_20: {
    base_model: "stabilityai/stable-diffusion-2",
    resolution: "768",
    format: "KOHYA_LORA",
    is_dit: false,
    has_te2: false,
    unet_dtype: "FLOAT_16",
    te_dtype: "FLOAT_16",
    timestep_distribution: "UNIFORM",
    sample_prompt: "a stunning view of the northern lights above a frozen lake"
  },
  STABLE_DIFFUSION_3: {
    base_model: "stabilityai/stable-diffusion-3-medium-diffusers",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: true,
    transformer_dtype: "FLOAT_16",
    te_dtype: "FLOAT_16",
    te2_dtype: "FLOAT_8",
    timestep_distribution: "LOGIT_NORMAL",
    sample_prompt: "a close up portrait of a futuristic astronaut with reflections in the visor, high detail, 8k"
  },
  STABLE_DIFFUSION_35: {
    base_model: "stabilityai/stable-diffusion-3.5-large",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: true,
    transformer_dtype: "INT_W8A8",
    te_dtype: "FLOAT_16",
    te2_dtype: "FLOAT_8",
    timestep_distribution: "LOGIT_NORMAL",
    sample_prompt: "a close up portrait of a futuristic astronaut with reflections in the visor, high detail, 8k"
  },
  SANA: {
    base_model: "Efficient-Large-Model/Sana_1600M_1024px",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: false,
    transformer_dtype: "BFLOAT_16",
    te_dtype: "BFLOAT_16",
    timestep_distribution: "LOGIT_NORMAL",
    sample_prompt: "artistic digital illustration of an enchanted forest with bioluminescent mushrooms, vibrant colors"
  },
  HUNYUAN_VIDEO: {
    base_model: "tencent/HunyuanVideo",
    resolution: "720",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: true,
    transformer_dtype: "INT_W8A8",
    te_dtype: "FLOAT_16",
    te2_dtype: "FLOAT_8",
    timestep_distribution: "LOGIT_NORMAL",
    sample_prompt: "a cinematic drone shot moving through a mountain valley with waterfalls, golden sunlight"
  },
  QWEN: {
    base_model: "Qwen/Qwen2-VL-7B-Instruct",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: false,
    transformer_dtype: "INT_W8A8",
    te_dtype: "FLOAT_8",
    timestep_distribution: "LOGIT_NORMAL",
    sample_prompt: "a high quality portrait of a student in a library reading a book, soft lighting"
  },
  ANIMA: {
    base_model: "cirno-ai/Anima-7B",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: false,
    transformer_dtype: "INT_W8A8",
    te_dtype: "FLOAT_8",
    timestep_distribution: "LOGIT_NORMAL",
    sample_prompt: "masterpiece anime style illustration of a magical girl with glowing staff"
  },
  Z_IMAGE: {
    base_model: "Z-Image-Ai/Z-Image",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: false,
    transformer_dtype: "INT_W8A8",
    te_dtype: "FLOAT_8",
    timestep_distribution: "LOGIT_NORMAL",
    sample_prompt: "stunning high-res landscape photograph with mountain range and clear reflection"
  },
  PIXART_SIGMA: {
    base_model: "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: false,
    transformer_dtype: "FLOAT_16",
    te_dtype: "FLOAT_16",
    timestep_distribution: "LOGIT_NORMAL",
    sample_prompt: "a detailed digital artwork of a mythical creature in a misty valley"
  },
  PIXART_ALPHA: {
    base_model: "PixArt-alpha/PixArt-XL-2-1024-MS",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: true,
    has_te2: false,
    transformer_dtype: "FLOAT_16",
    te_dtype: "FLOAT_16",
    timestep_distribution: "UNIFORM",
    sample_prompt: "a majestic mountain landscape at sunrise, 8k wallpaper"
  },
  STABLE_CASCADE_1: {
    base_model: "stabilityai/stable-cascade",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: false,
    has_te2: false,
    unet_dtype: "FLOAT_16",
    te_dtype: "FLOAT_16",
    timestep_distribution: "UNIFORM",
    sample_prompt: "an intricate steampunk clockwork mechanism with brass gears and steam"
  },
  WUERSTCHEN_2: {
    base_model: "warp-ai/wuerstchen",
    resolution: "1024",
    format: "DIFFUSERS_LORA",
    is_dit: false,
    has_te2: false,
    unet_dtype: "FLOAT_16",
    te_dtype: "FLOAT_16",
    timestep_distribution: "UNIFORM",
    sample_prompt: "a vibrant impressionist oil painting of a bustling harbor in summer"
  }
};

let currentConfig = {};
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

// Update UI visibility and defaults when Model Architecture changes
function applyArchitectureDefaults(archType) {
  const meta = ARCHITECTURE_DEFAULTS[archType];
  if (!meta) return;

  // 1. Auto-update Base Model Path
  const baseModelInput = document.getElementById('base_model_name');
  if (baseModelInput) {
    baseModelInput.value = meta.base_model;
    setNestedValue(currentConfig, 'base_model_name', meta.base_model);
  }

  // 2. Auto-update Output Format
  const outputFormatSelect = document.getElementById('output_model_format');
  if (outputFormatSelect && meta.format) {
    outputFormatSelect.value = meta.format;
    setNestedValue(currentConfig, 'output_model_format', meta.format);
  }

  // 3. Auto-update Output Destination Filename
  const destInput = document.getElementById('output_model_destination');
  if (destInput) {
    const safeArch = archType.toLowerCase().replace(/\s+/g, '_');
    const defaultDest = `output/${safeArch}_lora.safetensors`;
    destInput.value = defaultDest;
    setNestedValue(currentConfig, 'output_model_destination', defaultDest);
  }

  // 4. Auto-update Timestep Distribution
  const distSelect = document.getElementById('timestep_distribution');
  if (distSelect && meta.timestep_distribution) {
    distSelect.value = meta.timestep_distribution;
    setNestedValue(currentConfig, 'timestep_distribution', meta.timestep_distribution);
  }

  // 5. Auto-update Sample Prompt
  const sampleTextArea = document.getElementById('sample_prompts');
  if (sampleTextArea && meta.sample_prompt) {
    sampleTextArea.value = meta.sample_prompt;
  }

  // 6. Dynamic Visibility: DiT (Transformer) vs UNet fields
  const transformerField = document.getElementById('transformer_dtype')?.closest('.form-group');
  const unetField = document.getElementById('unet_dtype')?.closest('.form-group');
  const offloadField = document.getElementById('transformer_offload')?.closest('.form-group');

  if (meta.is_dit) {
    if (transformerField) transformerField.style.display = 'flex';
    if (offloadField) offloadField.style.display = 'flex';
    if (unetField) unetField.style.display = 'none';
  } else {
    if (transformerField) transformerField.style.display = 'none';
    if (offloadField) offloadField.style.display = 'none';
    if (unetField) unetField.style.display = 'flex';
  }

  // 7. Update Text Encoder 2 visibility
  const te2Field = document.getElementById('te2_dtype')?.closest('.form-group');
  if (te2Field) {
    te2Field.style.display = meta.has_te2 ? 'flex' : 'none';
  }
}

// Listen to Model Architecture dropdown changes
document.getElementById('model_type').addEventListener('change', (e) => {
  const chosenArch = e.target.value;
  applyArchitectureDefaults(chosenArch);
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
    populateForm(currentConfig);
    if (currentConfig.model_type) {
      applyArchitectureDefaults(currentConfig.model_type);
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
    container.innerHTML = '<div class="empty-placeholder">No concepts configured. Click "1-Click Auto-Detect" or "+ Add Concept Manually" above.</div>';
    return;
  }

  concepts.forEach((c, idx) => {
    const card = document.createElement('div');
    card.className = 'concept-card';
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
          <input type="text" id="concept_img_${idx}" value="${c.image_folder || ''}" onchange="updateConcept(${idx}, 'image_folder', this.value)" placeholder="/workspace/datasets/my_concept" />
          <button class="btn btn-secondary btn-browse" onclick="openFileBrowser('concept_img_${idx}')">📁</button>
        </div>
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

// 1-Click VRAM Hardware Optimizer
window.applyVRAMProfile = async function(vramGb) {
  try {
    syncFormToConfig();
    const res = await fetch('/api/autotune_vram', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vram_gb: vramGb })
    });
    const data = await res.json();
    if (data.status === 'ok') {
      currentConfig = data.config;
      populateForm(currentConfig);
      if (currentConfig.model_type) {
        applyArchitectureDefaults(currentConfig.model_type);
      }
      alert(`⚡ Applied ${vramGb}GB VRAM training profile! Hyperparameters, quantization, and offloading adjusted.`);
    }
  } catch (e) {
    alert(`Failed to apply VRAM profile: ${e}`);
  }
};

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

      // Smart Defaults & UI State sync
      const modelType = currentConfig.model_type;
      if (modelType) {
        applyArchitectureDefaults(modelType);
      }

      // Populate forms
      populateForm(currentConfig);

      // Auto-name output destination if default
      const safeCat = cat.toLowerCase().replace(/\s+/g, '_');
      const safeName = file.replace(/#/g, '').replace(/\.json/g, '').toLowerCase().replace(/\s+/g, '_');
      const targetDest = `output/${safeCat}_${safeName}.safetensors`;
      document.getElementById('output_model_destination').value = targetDest;
      setNestedValue(currentConfig, 'output_model_destination', targetDest);

      renderConcepts(currentConfig.concepts || []);
      alert(`Loaded preset: ${cat} (${file})! All parameters, base model repo, and precision updated.`);
    }
  } catch (err) {
    alert(`Failed to load preset: ${err}`);
  }
});

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
  applyArchitectureDefaults(modelType);
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
        applyArchitectureDefaults(currentConfig.model_type);
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
  fetchConfig();
  fetchPresets();
});
