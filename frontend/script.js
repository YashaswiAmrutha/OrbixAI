const API_BASE = window.location.protocol === "file:"
  ? "http://127.0.0.1:8001"
  : "";

let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let voiceMuted = false;
let emailRefreshInterval = null;
let healthCheckInterval = null;

const chatContainer = document.getElementById("chat-container");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const micBtn = document.getElementById("mic-btn");
const voiceMuteBtn = document.getElementById("voice-mute-btn");

const themeToggleBtn = document.getElementById("theme-toggle");

const mailContainer = document.getElementById("mail-container");
const mailList = document.getElementById("mail-list");
const mailRefreshBtn = document.getElementById("mail-refresh-btn");
const mailToggleBtn = document.getElementById("mail-toggle-btn");
const mailModal = document.getElementById("mail-modal");
const mailForm = document.getElementById("mail-form");
const modalOverlay = document.getElementById("modal-overlay");
const modalClose = document.getElementById("modal-close");
const modalCancel = document.getElementById("modal-cancel");

function initializeTheme() {
  const savedTheme = localStorage.getItem("theme");
  if (savedTheme === "light") {
    document.body.classList.add("light-mode");
  } else {
    document.body.classList.remove("light-mode");
  }

  const savedVoiceMute = localStorage.getItem("voiceMuted");
  if (savedVoiceMute === "true") {
    voiceMuted = true;
    if (voiceMuteBtn) voiceMuteBtn.classList.add("muted");
  }
}

function toggleTheme() {
  document.body.classList.toggle("light-mode");
  const isLightMode = document.body.classList.contains("light-mode");
  localStorage.setItem("theme", isLightMode ? "light" : "dark");
}

function toggleVoiceMute() {
  voiceMuted = !voiceMuted;
  localStorage.setItem("voiceMuted", voiceMuted);
  if (voiceMuteBtn) voiceMuteBtn.classList.toggle("muted");
}

if (themeToggleBtn) themeToggleBtn.addEventListener("click", toggleTheme);
if (voiceMuteBtn) voiceMuteBtn.addEventListener("click", toggleVoiceMute);

if (sendBtn) sendBtn.addEventListener("click", sendMessage);

if (userInput) userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

if (micBtn) micBtn.addEventListener("click", startVoice);

if (mailRefreshBtn) mailRefreshBtn.addEventListener("click", () => fetchEmails(true));
if (mailToggleBtn) mailToggleBtn.addEventListener("click", toggleMailWidget);
if (modalClose) modalClose.addEventListener("click", closeMailModal);
if (modalCancel) modalCancel.addEventListener("click", closeMailModal);
if (modalOverlay) modalOverlay.addEventListener("click", closeMailModal);
if (mailForm) mailForm.addEventListener("submit", handleMailSubmit);

function scrollToBottom() {
  setTimeout(() => {
    if (chatContainer) chatContainer.scrollTop = chatContainer.scrollHeight;
  }, 50);
}

function renderMarkdown(text) {
  // Escape HTML first
  let s = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  // ## Heading
  s = s.replace(/^## (.+)$/gm, '<strong style="font-size:15px;color:var(--txt1)">$1</strong>');
  // ### Heading
  s = s.replace(/^### (.+)$/gm, '<strong style="font-size:14px;color:var(--txt1)">$1</strong>');
  // **bold**
  s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // *italic*
  s = s.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // `code`
  s = s.replace(/`([^`]+)`/g, '<code style="background:rgba(59,130,246,0.12);padding:1px 5px;border-radius:4px;font-size:12px">$1</code>');
  // Bare URLs → clickable links
  s = s.replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1" target="_blank" style="color:var(--accent);text-decoration:underline">$1</a>');
  return s;
}

function addMessage(text, role) {
  if (!chatContainer) return;
  const wrapper = document.createElement("div");
  wrapper.className = `message-wrapper ${role}`;

  const messageDiv = document.createElement("div");
  messageDiv.className = "message";
  if (role === "assistant") {
    messageDiv.innerHTML = renderMarkdown(text);
  } else {
    messageDiv.textContent = text;
  }

  wrapper.appendChild(messageDiv);
  chatContainer.appendChild(wrapper);
  scrollToBottom();
}

function addStatusMessage(text) {
  if (!chatContainer) return null;
  const wrapper = document.createElement("div");
  wrapper.className = "message-wrapper status";

  const messageDiv = document.createElement("div");
  messageDiv.className = "message status-msg";
  messageDiv.textContent = text;

  wrapper.appendChild(messageDiv);
  chatContainer.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

function removeElement(el) {
  if (el && el.parentNode) {
    el.parentNode.removeChild(el);
  }
}

// ── Thinking bubble ─────────────────────────────────────────────────────────
let _thinkingEl = null;
let _thinkStart = 0;

function showThinking() {
  if (_thinkingEl) removeElement(_thinkingEl);
  _thinkStart = Date.now();

  const wrapper = document.createElement("div");
  wrapper.className = "message-wrapper assistant thinking-wrapper";

  wrapper.innerHTML = `
    <div class="thinking-bubble" id="thinking-bubble">
      <div class="thinking-header" onclick="this.closest('.thinking-bubble').classList.toggle('collapsed')">
        <span class="thinking-spinner"></span>
        <span class="thinking-label">Thinking…</span>
        <svg class="thinking-chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
      </div>
      <div class="thinking-steps" id="thinking-steps"></div>
    </div>`;

  chatContainer.appendChild(wrapper);
  _thinkingEl = wrapper;
  scrollToBottom();
  return wrapper;
}

function addThinkingStep(step) {
  const steps = document.getElementById("thinking-steps");
  if (!steps) return;
  const div = document.createElement("div");
  div.className = "thinking-step";
  div.textContent = step;
  // mark previous steps as done
  steps.querySelectorAll(".thinking-step.current").forEach(el => {
    el.classList.remove("current");
    el.classList.add("done");
  });
  div.classList.add("current");
  steps.appendChild(div);
  scrollToBottom();
}

function collapseThinking() {
  if (!_thinkingEl) return;
  const secs = ((Date.now() - _thinkStart) / 1000).toFixed(1);
  const bubble = _thinkingEl.querySelector(".thinking-bubble");
  if (bubble) {
    bubble.classList.add("collapsed", "done");
    const label = bubble.querySelector(".thinking-label");
    if (label) label.textContent = `Thought for ${secs}s`;
  }
  _thinkingEl = null;
}

async function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return;

  addMessage(message, "user");
  userInput.value = "";
  sendBtn.disabled = true;

  showThinking();

  try {
    const res = await fetch(`${API_BASE}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    });

    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE lines end with \n\n
      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // keep incomplete chunk

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        let evt;
        try { evt = JSON.parse(line.slice(5).trim()); }
        catch { continue; }

        if (evt.type === "thinking") {
          addThinkingStep(evt.step);

        } else if (evt.type === "response") {
          collapseThinking();

          if (evt.intent && evt.intent !== "general_chat") {
            const label = evt.intent.replace(/_/g, " ");
            addStatusMessage(`Intent: ${label}`);
          }

          if (evt.action === "open_mail_modal") {
            if (evt.reply) { addMessage(evt.reply, "assistant"); speak(evt.reply); }
            openMailModal(evt.parameters || {});
          } else if (evt.reply) {
            addMessage(evt.reply, "assistant");
            speak(evt.reply);
          }

        } else if (evt.type === "error") {
          collapseThinking();
          addMessage("Error: " + evt.message, "assistant");
        }
      }
    }
  } catch (error) {
    collapseThinking();
    addMessage("Connection error. Please check if the backend is running.", "assistant");
  } finally {
    sendBtn.disabled = false;
    if (userInput) userInput.focus();
  }
}

async function startVoice() {
  if (isRecording) {
    mediaRecorder.stop();
    micBtn.classList.remove("recording");
    isRecording = false;
    return;
  }

  try {
    audioChunks = [];
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    micBtn.classList.add("recording");
    isRecording = true;

    mediaRecorder.ondataavailable = (event) => {
      audioChunks.push(event.data);
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
      const formData = new FormData();
      formData.append("file", audioBlob, "voice.wav");

      showThinking();
      addThinkingStep("Processing voice…");

      try {
        const response = await fetch(`${API_BASE}/voice`, {
          method: "POST",
          body: formData
        });

        const data = await response.json();
        collapseThinking();

        if (data.intent) {
          addStatusMessage(`Intent: ${data.intent.replace(/_/g, " ")}`);
        }
        if (data.text) {
          addMessage(data.text, "user");
        }
        if (data.reply) {
          addMessage(data.reply, "assistant");
          speak(data.reply);
        }
      } catch (error) {
        collapseThinking();
        addMessage("Error processing voice. Please try again.", "assistant");
      }

      stream.getTracks().forEach((track) => track.stop());
    };

    mediaRecorder.start();

    setTimeout(() => {
      if (isRecording && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
        micBtn.classList.remove("recording");
        isRecording = false;
      }
    }, 10000);
  } catch (error) {
    addMessage("Microphone access denied. Please check permissions.", "assistant");
    micBtn.classList.remove("recording");
    isRecording = false;
  }
}

function speak(text) {
  if (voiceMuted || !("speechSynthesis" in window)) {
    return;
  }

  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1;
  utterance.pitch = 1;
  utterance.volume = 1;

  window.speechSynthesis.speak(utterance);
}

async function fetchEmails(manual) {
  const refreshIcon = mailRefreshBtn ? mailRefreshBtn.querySelector("svg") : null;

  if (manual && refreshIcon) {
    refreshIcon.classList.add("spin-icon");
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);

  try {
    const response = await fetch(`${API_BASE}/emails/latest?max_results=10`, {
      signal: controller.signal
    });

    clearTimeout(timeout);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    console.log("[OrbixAI] /emails/latest response:", JSON.stringify(data));

    if (data.error === "needs_auth") {
      // Auto-redirect to Google login
      console.log("[OrbixAI] Auth needed, redirecting to /auth/login");
      stopEmailAutoRefresh();
      window.location.href = `${API_BASE}/auth/login`;
      return;
    }

    const emailsToDisplay = data.emails || [];

    if (data.error) {
      displayEmailError(data.error);
    } else if (emailsToDisplay.length === 0) {
      displayEmails([]);
    } else {
      displayEmails(emailsToDisplay);
    }
  } catch (error) {
    clearTimeout(timeout);
    console.error("[OrbixAI] fetchEmails error:", error);
    if (error.name === "AbortError") {
      displayEmailError("Connection timeout. Retrying\u2026");
    } else {
      displayEmailError("Cannot reach server");
    }
  } finally {
    if (refreshIcon) {
      setTimeout(() => refreshIcon.classList.remove("spin-icon"), 600);
    }
  }
}

function startEmailAutoRefresh() {
  fetchEmails(false);

  emailRefreshInterval = setInterval(() => {
    fetchEmails(false);
  }, 30000);
}

function stopEmailAutoRefresh() {
  if (emailRefreshInterval) {
    clearInterval(emailRefreshInterval);
    emailRefreshInterval = null;
  }
}

function displayEmails(emails) {
  if (!mailList) return;
  if (emails.length === 0) {
    mailList.innerHTML = '<div class="mail-item placeholder"><p>No emails</p></div>';
    return;
  }

  mailList.innerHTML = "";
  emails.forEach(email => {
    const mailItem = document.createElement("div");
    mailItem.className = "mail-item";

    const badge = email.type === "sent" ? "sent" : "received";

    mailItem.innerHTML = `
      <div class="mail-icon-indicator ${badge}"></div>
      <div class="mail-content">
        <div class="mail-from">${email.from}</div>
        <div class="mail-subject">${email.subject || "No Subject"}</div>
      </div>
      <div class="mail-type-badge ${badge}">${email.type}</div>
    `;

    mailItem.addEventListener("click", () => {
      const snippet = email.snippet || email.body || "";
      const message = `Email from ${email.from}:\nSubject: ${email.subject}\n\n${snippet.substring(0, 200)}...`;
      addMessage(message, "assistant");
    });

    mailList.appendChild(mailItem);
  });
}

function displayEmailError(errorMsg) {
  if (!mailList) return;
  let friendlyMsg = "Unable to load emails";
  if (errorMsg.includes("not initialized")) {
    friendlyMsg = "Gmail not connected. Check credentials.";
  } else if (errorMsg.includes("timed out") || errorMsg.includes("timeout")) {
    friendlyMsg = "Gmail timed out. Will retry\u2026";
  } else if (errorMsg.includes("server") || errorMsg.includes("Server")) {
    friendlyMsg = "Server unreachable";
  }
  mailList.innerHTML = `<div class="mail-item placeholder error-state">
    <p>${friendlyMsg}</p>
    <button class="mail-retry-btn" onclick="fetchEmails(true)">Retry</button>
  </div>`;
}

function openMailModal(prefill) {
  if (!prefill) prefill = {};

  const recipientInput = document.getElementById("recipient-email");
  const promptInput    = document.getElementById("mail-prompt");

  if (recipientInput && (prefill.recipient_email || prefill.attendee_email)) {
    recipientInput.value = prefill.recipient_email || prefill.attendee_email;
  }

  // Pre-fill prompt with orchestrator context
  if (promptInput) {
    const emailContent = prefill.email_content || {};
    if (emailContent.subject || emailContent.body) {
      // Show subject + body preview in the prompt textarea
      const parts = [];
      if (emailContent.subject) parts.push(`Subject: ${emailContent.subject}`);
      if (emailContent.body)    parts.push(emailContent.body);
      promptInput.value = parts.join("\n\n");
      // Switch off LLM generation since content is already ready
      const useLLMCheck = document.getElementById("use-llm");
      if (useLLMCheck) useLLMCheck.checked = false;
    } else if (prefill.event_title) {
      promptInput.value = `Meeting: ${prefill.event_title}${prefill.event_description ? "\n" + prefill.event_description : ""}`;
    }
  }

  if (mailModal) mailModal.classList.add("active");
  if (modalOverlay) modalOverlay.classList.add("active");
  document.body.style.overflow = "hidden";
}

function closeMailModal() {
  if (mailModal) mailModal.classList.remove("active");
  if (modalOverlay) modalOverlay.classList.remove("active");
  document.body.style.overflow = "";
  if (mailForm) mailForm.reset();
}

function toggleMailWidget() {
  if (mailContainer) mailContainer.classList.toggle("collapsed");
}

async function handleMailSubmit(e) {
  e.preventDefault();

  const toEmail    = document.getElementById("recipient-email").value;
  const mailPrompt = document.getElementById("mail-prompt").value;
  const createMeet = document.getElementById("create-meet").checked;
  const useLLM     = document.getElementById("use-llm").checked;

  if (!toEmail) {
    addMessage("Please enter a recipient email address.", "assistant");
    return;
  }
  if (!mailPrompt) {
    addMessage("Please describe the email purpose or provide subject/body.", "assistant");
    return;
  }

  // If the prompt contains pre-filled "Subject: ..." content, extract it
  let extractedSubject = "";
  let extractedBody    = "";
  if (!useLLM && mailPrompt.includes("Subject:")) {
    const lines = mailPrompt.split("\n");
    const subjectLine = lines.find(l => l.startsWith("Subject:"));
    if (subjectLine) {
      extractedSubject = subjectLine.replace("Subject:", "").trim();
      extractedBody    = lines.slice(lines.indexOf(subjectLine) + 1).join("\n").replace(/^\s*\n/, "").trim();
    }
  }

  closeMailModal();

  try {
    let meetLink = null;

    // Step 1: Create Google Meet if requested
    if (createMeet) {
      addStatusMessage("Creating Google Meet\u2026");

      const meetRes = await fetch(`${API_BASE}/meetings/create`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          event_title: "Meeting with OrbixAI",
          event_description: mailPrompt,
          attendee_email: toEmail,
          send_email: false,
          user_prompt: mailPrompt
        })
      });

      const meetData = await meetRes.json();

      if (meetData.success) {
        meetLink = meetData.meet_link;
        addMessage(`Google Meet created: ${meetLink}`, "assistant");
      } else {
        addMessage(`Failed to create Google Meet: ${meetData.error || "Unknown error"}`, "assistant");
      }
    }

    // Step 2: Send email
    addStatusMessage("Sending email\u2026");

    const emailPayload = {
      to_email:       toEmail,
      use_llm:        useLLM && !extractedSubject,
      user_prompt:    mailPrompt,
      recipient_name: toEmail.split("@")[0]
    };

    if (meetLink) emailPayload.meeting_link = meetLink;

    // Use orchestrator-extracted subject/body if present
    if (extractedSubject) {
      emailPayload.subject = extractedSubject;
      emailPayload.body    = extractedBody + (meetLink ? `\n\nGoogle Meet Link: ${meetLink}` : "");
    } else if (!useLLM) {
      emailPayload.subject = createMeet ? "Meeting Invitation" : "Message from OrbixAI";
      emailPayload.body    = mailPrompt + (meetLink ? `\n\nGoogle Meet Link: ${meetLink}` : "");
    }

    const emailRes = await fetch(`${API_BASE}/emails/send`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(emailPayload)
    });

    const emailData = await emailRes.json();

    if (emailData.success) {
      addMessage(`Email sent successfully to ${toEmail}`, "assistant");
    } else {
      addMessage(`Failed to send email: ${emailData.error || "Unknown error"}`, "assistant");
    }

    setTimeout(() => fetchEmails(false), 1000);
  } catch (error) {
    addMessage(`Error: ${error.message}`, "assistant");
  }
}

// ===== Backend Health Check =====
async function checkBackendHealth() {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);

  try {
    const response = await fetch(`${API_BASE}/health`, { signal: controller.signal });
    clearTimeout(timeout);
  } catch (error) {
    clearTimeout(timeout);
  }
}

function startHealthCheck() {
  checkBackendHealth();
  healthCheckInterval = setInterval(checkBackendHealth, 15000);
}

function stopHealthCheck() {
  if (healthCheckInterval) {
    clearInterval(healthCheckInterval);
    healthCheckInterval = null;
  }
}

// ===== MINI CALENDAR =====
let calDate = new Date();

function renderCalendar() {
  const label = document.getElementById("cal-month-label");
  const grid  = document.getElementById("cal-grid");
  if (!label || !grid) return;

  const year  = calDate.getFullYear();
  const month = calDate.getMonth();
  const today = new Date();

  label.textContent = calDate.toLocaleString("default", { month: "long", year: "numeric" });

  const dayNames = ["Su","Mo","Tu","We","Th","Fr","Sa"];
  let html = dayNames.map(d => `<div class="cal-day-name">${d}</div>`).join("");

  const firstDay = new Date(year, month, 1).getDay();
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const daysInPrev  = new Date(year, month, 0).getDate();

  for (let i = 0; i < firstDay; i++) {
    html += `<div class="cal-day other-month"><span>${daysInPrev - firstDay + 1 + i}</span></div>`;
  }
  for (let d = 1; d <= daysInMonth; d++) {
    const isToday = d === today.getDate() && month === today.getMonth() && year === today.getFullYear();
    html += `<div class="cal-day${isToday ? " today" : ""}"><span>${d}</span></div>`;
  }
  grid.innerHTML = html;
}

function initCalendar() {
  renderCalendar();
  const prev = document.getElementById("cal-prev");
  const next = document.getElementById("cal-next");
  if (prev) prev.addEventListener("click", () => { calDate.setMonth(calDate.getMonth() - 1); renderCalendar(); });
  if (next) next.addEventListener("click", () => { calDate.setMonth(calDate.getMonth() + 1); renderCalendar(); });
}

function initTodayInfo() {
  const now = new Date();
  const dateEl = document.getElementById("today-date");
  const dayEl  = document.getElementById("today-day");
  if (dateEl) dateEl.textContent = now.getDate();
  if (dayEl)  dayEl.textContent  = now.toLocaleString("default", { weekday: "long", month: "short", year: "numeric" });
}

// ===== TO-DO LIST =====
let todos = JSON.parse(localStorage.getItem("orbix-todos") || "[]");

function saveTodos() {
  localStorage.setItem("orbix-todos", JSON.stringify(todos));
}

function renderTodos() {
  const list      = document.getElementById("todo-list");
  const countEl   = document.getElementById("todo-count");
  if (!list) return;

  const pending = todos.filter(t => !t.done).length;
  if (countEl) countEl.textContent = pending;

  if (todos.length === 0) {
    list.innerHTML = `<div class="todo-empty">No tasks yet</div>`;
    return;
  }

  list.innerHTML = todos.map((t, i) => `
    <div class="todo-item">
      <div class="todo-check ${t.done ? "done" : ""}" data-idx="${i}"></div>
      <span class="todo-text ${t.done ? "done" : ""}">${escapeHtml(t.text)}</span>
      <button class="todo-del-btn" data-idx="${i}">✕</button>
    </div>
  `).join("");

  list.querySelectorAll(".todo-check").forEach(el => {
    el.addEventListener("click", () => {
      todos[+el.dataset.idx].done = !todos[+el.dataset.idx].done;
      saveTodos();
      renderTodos();
    });
  });
  list.querySelectorAll(".todo-del-btn").forEach(el => {
    el.addEventListener("click", () => {
      todos.splice(+el.dataset.idx, 1);
      saveTodos();
      renderTodos();
    });
  });
}

function escapeHtml(str) {
  return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function initTodos() {
  renderTodos();
  const input  = document.getElementById("todo-input");
  const addBtn = document.getElementById("todo-add-btn");

  function addTodo() {
    if (!input) return;
    const text = input.value.trim();
    if (!text) return;
    todos.unshift({ text, done: false });
    saveTodos();
    renderTodos();
    input.value = "";
    input.focus();
  }

  if (addBtn) addBtn.addEventListener("click", addTodo);
  if (input)  input.addEventListener("keypress", e => { if (e.key === "Enter") addTodo(); });
}

window.addEventListener("load", () => {
  initializeTheme();
  if (userInput) userInput.focus();
  addStatusMessage("Welcome to OrbixAI. Start chatting or use voice.");
  startEmailAutoRefresh();
  startHealthCheck();
  initCalendar();
  initTodayInfo();
  initTodos();
});

window.addEventListener("beforeunload", () => {
  stopEmailAutoRefresh();
  stopHealthCheck();
});
