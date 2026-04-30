/* ===== INF232 EC2 — charts.js ===== */
document.addEventListener('DOMContentLoaded', function () {

  // Upload zone drag & drop
  const uploadZone = document.getElementById('uploadZone');
  const fileInput  = document.getElementById('fileInput');
  if (uploadZone && fileInput) {
    uploadZone.addEventListener('click', () => fileInput.click());
    uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
    uploadZone.addEventListener('drop', e => {
      e.preventDefault(); uploadZone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) { fileInput.files = e.dataTransfer.files; updateFileLabel(e.dataTransfer.files[0].name); }
    });
    fileInput.addEventListener('change', () => { if (fileInput.files.length > 0) updateFileLabel(fileInput.files[0].name); });
    function updateFileLabel(name) {
      const label = document.getElementById('fileLabel');
      if (label) { label.innerHTML = '<i class="bi bi-file-earmark-check me-2 text-success"></i>' + name; }
    }
  }

  // Histogramme dynamique
  const histColSelect = document.getElementById('histColSelect');
  if (histColSelect) {
    histColSelect.addEventListener('change', function () {
      const col = this.value; if (!col) return;
      const img = document.getElementById('histPlotImg');
      const spin = document.getElementById('histLoading');
      if (spin) spin.classList.remove('d-none');
      fetch('/data/plot/histogram?col=' + encodeURIComponent(col))
        .then(r => r.json()).then(data => {
          if (data.plot && img) { img.src = 'data:image/png;base64,' + data.plot; img.style.opacity = '1'; }
          if (spin) spin.classList.add('d-none');
        }).catch(() => { if (spin) spin.classList.add('d-none'); });
    });
  }

  // Scatter dynamique
  const scatterBtn = document.getElementById('scatterBtn');
  if (scatterBtn) {
    scatterBtn.addEventListener('click', function () {
      const x = document.getElementById('scatterX')?.value;
      const y = document.getElementById('scatterY')?.value;
      const img = document.getElementById('scatterPlotImg');
      if (!x || !y || !img) return;
      img.style.opacity = '0.4';
      fetch('/data/plot/scatter?x=' + encodeURIComponent(x) + '&y=' + encodeURIComponent(y))
        .then(r => r.json()).then(data => { if (data.plot) { img.src = 'data:image/png;base64,' + data.plot; img.style.opacity = '1'; } });
    });
  }

  // Compteur sélection multiple
  document.querySelectorAll('select[multiple]').forEach(sel => {
    sel.addEventListener('change', function () {
      const badge = document.getElementById(this.id + 'Count');
      if (badge) badge.textContent = this.selectedOptions.length + ' sélectionné(s)';
    });
  });

  // Auto-dismiss alertes succès
  document.querySelectorAll('.alert.alert-success').forEach(a => {
    setTimeout(() => bootstrap.Alert.getOrCreateInstance(a)?.close(), 6000);
  });

  // Spinner sur submit
  document.querySelectorAll('form.needs-loading').forEach(form => {
    form.addEventListener('submit', function () {
      const btn = form.querySelector('button[type="submit"]');
      if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Calcul en cours...'; }
    });
  });

  // Tooltips Bootstrap
  document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => new bootstrap.Tooltip(el));
});
