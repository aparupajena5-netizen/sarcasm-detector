const contextEl = document.getElementById('context');
const commentEl = document.getElementById('comment');
const analyzeBtn = document.getElementById('analyzeBtn');
const btnText = document.getElementById('btnText');
const clearBtn = document.getElementById('clearBtn');
const statusText = document.getElementById('statusText');
const scoreEl = document.getElementById('score');
const probabilityEl = document.getElementById('probability');
const scoreRing = document.getElementById('scoreRing');
const barFill = document.getElementById('barFill');
const labelBadge = document.getElementById('labelBadge');
const predictionTitle = document.getElementById('predictionTitle');
const predictionText = document.getElementById('predictionText');
const attentionWords = document.getElementById('attentionWords');
const rawResult = document.getElementById('rawResult');

function setResult(res) {
  const p = Number(res.prob || 0);
  const pct = Math.round(p * 100);
  const flag = Boolean(res.flag);
  scoreEl.textContent = p.toFixed(2);
  probabilityEl.textContent = `${pct}%`;
  barFill.style.width = `${pct}%`;
  scoreRing.style.background = `conic-gradient(var(--accent) ${pct * 3.6}deg, rgba(255,255,255,.06) ${pct * 3.6}deg)`;
  labelBadge.textContent = flag ? 'SARCASTIC' : 'NOT SARCASTIC';
  labelBadge.className = `badge ${flag ? 'sarcastic' : 'normal'}`;
  predictionTitle.textContent = flag ? 'Sarcasm detected' : 'No sarcasm detected';
  predictionText.textContent = flag
    ? 'The model considers this message likely sarcastic or passive-aggressive.'
    : 'The model did not cross its sarcasm decision threshold.';

  attentionWords.innerHTML = '';
  const words = Array.isArray(res.top_words) ? res.top_words : [];
  if (!words.length) {
    attentionWords.innerHTML = '<span class="empty">No attention words returned.</span>';
  } else {
    words.forEach(([word, weight]) => {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.innerHTML = `${escapeHtml(word)} <b>${Number(weight).toFixed(3)}</b>`;
      attentionWords.appendChild(chip);
    });
  }
  rawResult.textContent = `Analyzed ${new Date().toLocaleTimeString()} • threshold: ${res.threshold ?? 'model default'} • ${res.tokens?.length ?? 0} token(s)`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

async function analyze() {
  const comment = commentEl.value.trim();
  const context = contextEl.value.trim();
  if (!comment) {
    commentEl.focus();
    predictionTitle.textContent = 'Enter a comment';
    predictionText.textContent = 'Write a message first, then click Analyze message.';
    return;
  }
  analyzeBtn.disabled = true;
  btnText.textContent = 'Analyzing…';
  statusText.textContent = 'Running model';
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({comment, context})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Prediction failed');
    setResult(data);
    statusText.textContent = 'Model ready';
  } catch (err) {
    statusText.textContent = 'Backend error';
    rawResult.textContent = err.message + ' — make sure app.py is running and the trained checkpoint exists in outputs/.';
    predictionTitle.textContent = 'Could not analyze';
    predictionText.textContent = 'Start the Python frontend server and try again.';
  } finally {
    analyzeBtn.disabled = false;
    btnText.textContent = 'Analyze message';
  }
}

analyzeBtn.addEventListener('click', analyze);
clearBtn.addEventListener('click', () => {
  contextEl.value = '';
  commentEl.value = '';
  scoreEl.textContent = '--';
  probabilityEl.textContent = '--';
  barFill.style.width = '0%';
  scoreRing.style.background = 'conic-gradient(var(--accent) 0deg, rgba(255,255,255,.06) 0deg)';
  labelBadge.textContent = 'Waiting';
  labelBadge.className = 'badge';
  predictionTitle.textContent = 'Enter a message';
  predictionText.textContent = 'Your prediction will appear here after analysis.';
  attentionWords.innerHTML = '<span class="empty">No prediction yet</span>';
  rawResult.textContent = 'Ready for your first test.';
});

document.querySelectorAll('.example').forEach(btn => {
  btn.addEventListener('click', () => {
    contextEl.value = btn.dataset.context || '';
    commentEl.value = btn.dataset.comment || '';
    commentEl.focus();
  });
});

document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyze();
});
