const chat = document.getElementById('chat');
const form = document.getElementById('composer');
const input = document.getElementById('question-input');
const sendBtn = document.getElementById('send-btn');

function scrollToBottom() {
  chat.scrollTop = chat.scrollHeight;
}

function addUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'msg-row user';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  row.appendChild(bubble);
  chat.appendChild(row);
  scrollToBottom();
}

function addThinkingMessage() {
  const row = document.createElement('div');
  row.className = 'msg-row bot';
  row.id = 'thinking-row';
  const bubble = document.createElement('div');
  bubble.className = 'bubble thinking';
  bubble.textContent = 'Sochte hain...';
  row.appendChild(bubble);
  chat.appendChild(row);
  scrollToBottom();
}

function removeThinkingMessage() {
  const el = document.getElementById('thinking-row');
  if (el) el.remove();
}

function addBotMessage(answer, sources) {
  const row = document.createElement('div');
  row.className = 'msg-row bot';

  const wrap = document.createElement('div');
  wrap.className = 'bubble-wrap';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = answer;
  wrap.appendChild(bubble);

  if (sources && sources.length > 0) {
    const tabs = document.createElement('div');
    tabs.className = 'source-tabs';
    sources.forEach(s => {
      const tab = document.createElement('span');
      tab.className = 'source-tab';
      tab.textContent = `Unit ${s.unit_id} \u00b7 ${s.unit_title}`;
      tabs.appendChild(tab);
    });
    wrap.appendChild(tabs);
  }

  row.appendChild(wrap);
  chat.appendChild(row);
  scrollToBottom();
}

function addErrorMessage(text) {
  const row = document.createElement('div');
  row.className = 'msg-row bot';
  const bubble = document.createElement('div');
  bubble.className = 'bubble error';
  bubble.textContent = text;
  row.appendChild(bubble);
  chat.appendChild(row);
  scrollToBottom();
}

async function askQuestion(question) {
  addThinkingMessage();
  sendBtn.disabled = true;
  input.disabled = true;

  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });

    removeThinkingMessage();

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      addErrorMessage(data.detail || 'Something went wrong. Try again in a moment.');
      return;
    }

    const data = await res.json();
    addBotMessage(data.answer, data.sources);
  } catch (err) {
    removeThinkingMessage();
    addErrorMessage('Connection problem -- check your internet and try again.');
  } finally {
    sendBtn.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  addUserMessage(question);
  input.value = '';
  askQuestion(question);
});

// Register service worker for PWA install support
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/service-worker.js').catch(() => {});
  });
}
