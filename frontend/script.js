const API_BASE = window.location.protocol === "file:"
  ? "http://127.0.0.1:8001"
  : "";

// ── Core state ───────────────────────────────────────────────────────────────
let mediaRecorder;
let audioChunks  = [];
let isRecording  = false;
let voiceMuted   = false;
let emailRefreshInterval = null;
let healthCheckInterval  = null;
let emailFetchInProgress = false;   // prevent overlapping fetches
let emailsDisplayed      = false;   // suppress transient errors once emails load

// ── Calendar event state ─────────────────────────────────────────────────────
let calendarEvents = {};  // { "YYYY-MM-DD": [{id, title, date, time, description, type, source, color}, ...] }
let selectedCalDate = null;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const chatContainer  = document.getElementById("chat-container");
const userInput      = document.getElementById("user-input");
const sendBtn        = document.getElementById("send-btn");
const micBtn         = document.getElementById("mic-btn");
const voiceMuteBtn   = document.getElementById("voice-mute-btn");
const themeToggleBtn = document.getElementById("theme-toggle");

const mailContainer = document.getElementById("mail-container");
const mailList      = document.getElementById("mail-list");
const mailRefreshBtn = document.getElementById("mail-refresh-btn");
const mailToggleBtn  = document.getElementById("mail-toggle-btn");
const mailModal      = document.getElementById("mail-modal");
const mailForm       = document.getElementById("mail-form");
const modalOverlay   = document.getElementById("modal-overlay");
const modalClose     = document.getElementById("modal-close");
const modalCancel    = document.getElementById("modal-cancel");

// ── Theme / Voice ─────────────────────────────────────────────────────────────
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
  localStorage.setItem("theme", document.body.classList.contains("light-mode") ? "light" : "dark");
}

function toggleVoiceMute() {
  voiceMuted = !voiceMuted;
  localStorage.setItem("voiceMuted", voiceMuted);
  if (voiceMuteBtn) voiceMuteBtn.classList.toggle("muted");
  // Cancel any in-progress speech immediately when muting
  if (voiceMuted && "speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
}

if (themeToggleBtn) themeToggleBtn.addEventListener("click", toggleTheme);
if (voiceMuteBtn)   voiceMuteBtn.addEventListener("click", toggleVoiceMute);
if (sendBtn)        sendBtn.addEventListener("click", sendMessage);
if (userInput)      userInput.addEventListener("keypress", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
if (micBtn)         micBtn.addEventListener("click", startVoice);

if (mailRefreshBtn) mailRefreshBtn.addEventListener("click", () => fetchEmails(true));
if (mailToggleBtn)  mailToggleBtn.addEventListener("click", toggleMailWidget);
if (modalClose)     modalClose.addEventListener("click", closeMailModal);
if (modalCancel)    modalCancel.addEventListener("click", closeMailModal);
if (modalOverlay)   modalOverlay.addEventListener("click", closeMailModal);
if (mailForm)       mailForm.addEventListener("submit", handleMailSubmit);

// ── Utilities ─────────────────────────────────────────────────────────────────
function scrollToBottom() {
  setTimeout(() => { if (chatContainer) chatContainer.scrollTop = chatContainer.scrollHeight; }, 50);
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function renderMarkdown(text) {
  let s = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  s = s.replace(/^## (.+)$/gm, '<strong style="font-size:15px;color:var(--txt1)">$1</strong>');
  s = s.replace(/^### (.+)$/gm, '<strong style="font-size:14px;color:var(--txt1)">$1</strong>');
  s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/\*(.+?)\*/g, '<em>$1</em>');
  s = s.replace(/`([^`]+)`/g, '<code style="background:var(--surface);border:1px solid var(--border);padding:1px 5px;border-radius:5px;font-size:12px;font-family:\'JetBrains Mono\',monospace">$1</code>');
  s = s.replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1" target="_blank" style="color:var(--accent-text);text-decoration:underline">$1</a>');
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
  if (el && el.parentNode) el.parentNode.removeChild(el);
}

// ── Thinking bubble ──────────────────────────────────────────────────────────
let _thinkingEl  = null;
let _thinkStart  = 0;

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
  const secs   = ((Date.now() - _thinkStart) / 1000).toFixed(1);
  const bubble = _thinkingEl.querySelector(".thinking-bubble");
  if (bubble) {
    bubble.classList.add("collapsed", "done");
    const label = bubble.querySelector(".thinking-label");
    if (label) label.textContent = `Thought for ${secs}s`;
  }
  _thinkingEl = null;
}

// ── Chat / Send ───────────────────────────────────────────────────────────────
async function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return;

  const hero = document.getElementById("welcome-hero");
  if (hero) hero.remove();

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

      const parts = buffer.split("\n\n");
      buffer = parts.pop();

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

          // Auto-sync calendar events from AI response
          if (evt.calendar_events && evt.calendar_events.length > 0) {
            autoSyncCalendarEvents(evt.calendar_events);
            addStatusMessage(`Added ${evt.calendar_events.length} event(s) to your calendar`);
          }

          // Auto-add AI-generated todo items
          if (evt.todo_items && evt.todo_items.length > 0) {
            autoAddTodos(evt.todo_items);
          }

          if (evt.intent && evt.intent !== "general_chat") {
            addStatusMessage(`Intent: ${evt.intent.replace(/_/g, " ")}`);
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

// ── Voice ─────────────────────────────────────────────────────────────────────
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

    mediaRecorder.ondataavailable = (event) => { audioChunks.push(event.data); };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
      const formData  = new FormData();
      formData.append("file", audioBlob, "voice.wav");

      showThinking();
      addThinkingStep("Processing voice…");

      try {
        const response = await fetch(`${API_BASE}/voice`, { method: "POST", body: formData });
        const data = await response.json();
        collapseThinking();
        if (data.intent)  addStatusMessage(`Intent: ${data.intent.replace(/_/g, " ")}`);
        if (data.text)    addMessage(data.text, "user");
        if (data.reply)   { addMessage(data.reply, "assistant"); speak(data.reply); }
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
  if (voiceMuted || !("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate   = 1;
  utterance.pitch  = 1;
  utterance.volume = 1;
  window.speechSynthesis.speak(utterance);
}

// ── Emails ────────────────────────────────────────────────────────────────────
async function fetchEmails(manual) {
  // Don't pile up parallel fetches — skip auto-refreshes while one is in flight
  if (emailFetchInProgress && !manual) return;
  emailFetchInProgress = true;

  const refreshIcon = mailRefreshBtn ? mailRefreshBtn.querySelector("svg") : null;
  if (manual && refreshIcon) refreshIcon.classList.add("spin-icon");

  const controller = new AbortController();
  // Give the backend (30 s) + some headroom → 35 s client-side timeout
  const timeout = setTimeout(() => controller.abort(), 35000);

  try {
    const response = await fetch(`${API_BASE}/emails/latest?max_results=10`, { signal: controller.signal });
    clearTimeout(timeout);
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);

    const data = await response.json();
    if (data.error === "needs_auth") {
      stopEmailAutoRefresh();
      window.location.href = `${API_BASE}/auth/login`;
      return;
    }
    const emailsToDisplay = data.emails || [];
    if (data.error && emailsToDisplay.length === 0) {
      // Only show error if we have nothing to display yet
      if (!emailsDisplayed) displayEmailError(data.error);
    } else if (emailsToDisplay.length === 0) {
      displayEmails([]);
    } else {
      displayEmails(emailsToDisplay);
      emailsDisplayed = true;
    }
  } catch (error) {
    clearTimeout(timeout);
    if (error.name === "AbortError") {
      // Timeout — only show error UI if we have no emails currently shown
      if (!emailsDisplayed) displayEmailError("Loading emails\u2026 retrying shortly");
    } else {
      if (!emailsDisplayed) displayEmailError("Cannot reach server");
    }
  } finally {
    emailFetchInProgress = false;
    if (refreshIcon) setTimeout(() => refreshIcon.classList.remove("spin-icon"), 600);
  }
}

function startEmailAutoRefresh() {
  fetchEmails(false);
  emailRefreshInterval = setInterval(() => fetchEmails(false), 30000);
}

function stopEmailAutoRefresh() {
  if (emailRefreshInterval) { clearInterval(emailRefreshInterval); emailRefreshInterval = null; }
}

function displayEmails(emails) {
  if (!mailList) return;
  if (emails.length === 0) {
    // Don't wipe a real email list with "No emails" on a possibly-stale empty response
    if (!emailsDisplayed) mailList.innerHTML = '<div class="mail-item placeholder"><p>No emails</p></div>';
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
        <div class="mail-from">${escapeHtml(email.from)}</div>
        <div class="mail-subject">${escapeHtml(email.subject || "No Subject")}</div>
      </div>
      <div class="mail-type-badge ${badge}">${email.type}</div>`;
    mailItem.addEventListener("click", () => {
      const snippet  = email.snippet || email.body || "";
      const message = `Email from ${email.from}:\nSubject: ${email.subject}\n\n${snippet.substring(0, 200)}...`;
      addMessage(message, "assistant");
    });
    mailList.appendChild(mailItem);
  });
}

function displayEmailError(errorMsg) {
  if (!mailList) return;
  let friendlyMsg = "Unable to load emails";
  if (errorMsg.includes("not initialized"))                               friendlyMsg = "Gmail not connected. Check credentials.";
  else if (errorMsg.includes("timed out") || errorMsg.includes("timeout")) friendlyMsg = "Gmail timed out. Will retry\u2026";
  else if (errorMsg.includes("server") || errorMsg.includes("Server"))    friendlyMsg = "Server unreachable";
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
  if (promptInput) {
    const emailContent = prefill.email_content || {};
    if (emailContent.subject || emailContent.body) {
      const parts = [];
      if (emailContent.subject) parts.push(`Subject: ${emailContent.subject}`);
      if (emailContent.body)    parts.push(emailContent.body);
      promptInput.value = parts.join("\n\n");
      const useLLMCheck = document.getElementById("use-llm");
      if (useLLMCheck) useLLMCheck.checked = false;
    } else if (prefill.event_title) {
      promptInput.value = `Meeting: ${prefill.event_title}${prefill.event_description ? "\n" + prefill.event_description : ""}`;
    }
  }
  if (mailModal)    mailModal.classList.add("active");
  if (modalOverlay) modalOverlay.classList.add("active");
  document.body.style.overflow = "hidden";
}

function closeMailModal() {
  if (mailModal)    mailModal.classList.remove("active");
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

  if (!toEmail)    { addMessage("Please enter a recipient email address.", "assistant"); return; }
  if (!mailPrompt) { addMessage("Please describe the email purpose or provide subject/body.", "assistant"); return; }

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
    if (createMeet) {
      addStatusMessage("Creating Google Meet\u2026");
      const meetRes  = await fetch(`${API_BASE}/meetings/create`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ event_title: "Meeting with OrbixAI", event_description: mailPrompt, attendee_email: toEmail, send_email: false, user_prompt: mailPrompt })
      });
      const meetData = await meetRes.json();
      if (meetData.success) {
        meetLink = meetData.meet_link;
        addMessage(`Google Meet created: ${meetLink}`, "assistant");

        // Auto-create calendar event for manually triggered meeting
        const today = new Date().toISOString().slice(0, 10);
        try {
          const ceRes = await fetch(`${API_BASE}/calendar/events`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title: "Meeting with OrbixAI", date: today, description: `Meet: ${meetLink}\nAttendee: ${toEmail}`, type: "meeting", source: "manual" })
          });
          const ceData = await ceRes.json();
          if (ceData.success) autoSyncCalendarEvents([ceData.event]);
        } catch (_) { /* non-critical */ }
      } else {
        addMessage(`Failed to create Google Meet: ${meetData.error || "Unknown error"}`, "assistant");
      }
    }

    addStatusMessage("Sending email\u2026");
    const emailPayload = { to_email: toEmail, use_llm: useLLM && !extractedSubject, user_prompt: mailPrompt, recipient_name: toEmail.split("@")[0] };
    if (meetLink)        emailPayload.meeting_link = meetLink;
    if (extractedSubject) {
      emailPayload.subject = extractedSubject;
      emailPayload.body    = extractedBody + (meetLink ? `\n\nGoogle Meet Link: ${meetLink}` : "");
    } else if (!useLLM) {
      emailPayload.subject = createMeet ? "Meeting Invitation" : "Message from OrbixAI";
      emailPayload.body    = mailPrompt + (meetLink ? `\n\nGoogle Meet Link: ${meetLink}` : "");
    }
    const emailRes  = await fetch(`${API_BASE}/emails/send`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(emailPayload) });
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

// ── Backend Health Check ──────────────────────────────────────────────────────
async function checkBackendHealth() {
  const controller = new AbortController();
  const timeout    = setTimeout(() => controller.abort(), 5000);
  try { await fetch(`${API_BASE}/health`, { signal: controller.signal }); clearTimeout(timeout); }
  catch { clearTimeout(timeout); }
}

function startHealthCheck() {
  checkBackendHealth();
  healthCheckInterval = setInterval(checkBackendHealth, 15000);
}

function stopHealthCheck() {
  if (healthCheckInterval) { clearInterval(healthCheckInterval); healthCheckInterval = null; }
}

// ── CALENDAR ──────────────────────────────────────────────────────────────────
let calDate = new Date();

const EVENT_TYPE_COLORS = {
  meeting: "#3b82f6",
  travel:  "#22c55e",
  task:    "#f59e0b",
  general: "#8b5cf6",
};
const EVENT_TYPE_LABELS = {
  meeting: "Meeting",
  travel:  "Travel",
  task:    "Task",
  general: "General",
};

// Fetch events from backend and populate cache
async function fetchCalendarEvents(year, month) {
  try {
    const res = await fetch(`${API_BASE}/calendar/events?year=${year}&month=${month}`);
    if (!res.ok) return;
    const data = await res.json();

    // Remove stale entries for this month
    const prefix = `${year}-${String(month).padStart(2, "0")}`;
    Object.keys(calendarEvents).forEach(k => { if (k.startsWith(prefix)) delete calendarEvents[k]; });

    // Populate cache
    (data.events || []).forEach(ev => {
      if (!calendarEvents[ev.date]) calendarEvents[ev.date] = [];
      if (!calendarEvents[ev.date].find(e => e.id === ev.id)) calendarEvents[ev.date].push(ev);
    });
    renderCalendar();
  } catch (e) {
    console.warn("[OrbixAI] fetchCalendarEvents error:", e);
  }
}

// Merge AI-pushed events into cache and re-render
function autoSyncCalendarEvents(events) {
  (events || []).forEach(ev => {
    if (!ev.date) return;
    if (!calendarEvents[ev.date]) calendarEvents[ev.date] = [];
    if (!calendarEvents[ev.date].find(e => e.id === ev.id)) calendarEvents[ev.date].push(ev);
  });
  renderCalendar();
  if (selectedCalDate) {
    const needsUpdate = (events || []).some(ev => ev.date === selectedCalDate);
    if (needsUpdate) {
      renderDayEvents(selectedCalDate);
      updateDayEventsCount(selectedCalDate);
    }
  }
}

function renderCalendar() {
  const label = document.getElementById("cal-month-label");
  const grid  = document.getElementById("cal-grid");
  if (!label || !grid) return;

  const year  = calDate.getFullYear();
  const month = calDate.getMonth();
  const today = new Date();

  label.textContent = calDate.toLocaleString("default", { month: "long", year: "numeric" });

  const dayNames    = ["Su","Mo","Tu","We","Th","Fr","Sa"];
  let html = dayNames.map(d => `<div class="cal-day-name">${d}</div>`).join("");

  const firstDay    = new Date(year, month, 1).getDay();
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const daysInPrev  = new Date(year, month, 0).getDate();

  for (let i = 0; i < firstDay; i++) {
    html += `<div class="cal-day other-month"><span>${daysInPrev - firstDay + 1 + i}</span></div>`;
  }

  for (let d = 1; d <= daysInMonth; d++) {
    const isToday    = d === today.getDate() && month === today.getMonth() && year === today.getFullYear();
    const dateStr    = `${year}-${String(month + 1).padStart(2,"0")}-${String(d).padStart(2,"0")}`;
    const isSelected = dateStr === selectedCalDate;
    const dayEvs     = calendarEvents[dateStr] || [];
    const hasEvents  = dayEvs.length > 0;

    const dots = hasEvents
      ? `<div class="cal-event-dots">${dayEvs.slice(0, 3).map(ev =>
          `<div class="cal-event-dot" style="background:${ev.color || EVENT_TYPE_COLORS[ev.type] || "#8b5cf6"}"></div>`
        ).join("")}</div>`
      : "";

    html += `<div class="cal-day${isToday ? " today" : ""}${isSelected ? " selected" : ""}${hasEvents ? " has-events" : ""}" data-day="${d}">
      <span>${d}</span>${dots}
    </div>`;
  }

  grid.innerHTML = html;

  // Click handlers for day cells
  grid.querySelectorAll(".cal-day:not(.other-month)").forEach(cell => {
    cell.style.cursor = "pointer";
    cell.addEventListener("click", () => {
      const day = parseInt(cell.dataset.day);
      showDayPanel(year, month, day);
    });
  });

  // Keep the Today agenda in sync with the latest events
  renderTodayAgenda();
}

// Show day events panel for a given day
function showDayPanel(year, month, day) {
  const dateStr = `${year}-${String(month + 1).padStart(2,"0")}-${String(day).padStart(2,"0")}`;
  selectedCalDate = dateStr;

  const panel = document.getElementById("day-events-panel");
  if (panel) panel.style.display = "";

  // Highlight selected cell
  document.querySelectorAll(".cal-day.selected").forEach(el => el.classList.remove("selected"));
  const grid = document.getElementById("cal-grid");
  if (grid) {
    grid.querySelectorAll(".cal-day:not(.other-month)").forEach(cell => {
      if (parseInt(cell.dataset.day) === day) cell.classList.add("selected");
    });
  }

  updateDayEventsTitle(dateStr);
  updateDayEventsCount(dateStr);
  renderDayEvents(dateStr);
}

function hideDayPanel() {
  const panel = document.getElementById("day-events-panel");
  if (panel) panel.style.display = "none";
  selectedCalDate = null;
  document.querySelectorAll(".cal-day.selected").forEach(el => el.classList.remove("selected"));
}

function updateDayEventsTitle(dateStr) {
  const titleEl = document.getElementById("day-events-title");
  if (!titleEl) return;
  try {
    const d = new Date(dateStr + "T00:00:00");
    titleEl.textContent = d.toLocaleString("default", { weekday: "short", month: "short", day: "numeric" });
  } catch (_) {
    titleEl.textContent = dateStr;
  }
}

function updateDayEventsCount(dateStr) {
  const countEl = document.getElementById("day-events-count");
  if (countEl) countEl.textContent = (calendarEvents[dateStr] || []).length;
}

function renderDayEvents(dateStr) {
  const list = document.getElementById("day-events-list");
  if (!list) return;
  updateDayEventsCount(dateStr);

  const events = calendarEvents[dateStr] || [];
  if (events.length === 0) {
    list.innerHTML = `<div class="day-events-empty">No events. Add one below.</div>`;
    return;
  }

  list.innerHTML = events.map(ev => {
    const color     = ev.color || EVENT_TYPE_COLORS[ev.type] || "#8b5cf6";
    const label     = EVENT_TYPE_LABELS[ev.type] || ev.type;
    const bgColor   = hexToRgba(color, 0.12);
    const bdColor   = hexToRgba(color, 0.25);
    const timeStr   = ev.time ? `<span class="day-event-time">${escapeHtml(ev.time)}</span>` : "";
    const aiSource  = ev.source === "ai_travel" ? "AI Travel" : ev.source === "ai_meeting" ? "AI Meeting" : "";
    const aiTag     = aiSource ? `<span class="event-source-badge">${aiSource}</span>` : "";

    return `<div class="day-event-item" role="listitem" style="border-left-color:${color}" data-id="${ev.id}" data-date="${ev.date}">
      <div class="day-event-body">
        <span class="day-event-title" data-id="${ev.id}" data-date="${ev.date}" tabindex="0" role="button" aria-label="Edit ${escapeHtml(ev.title)}">${escapeHtml(ev.title)}</span>
        <div class="day-event-meta">
          ${timeStr}
          <span class="event-type-badge" style="background:${bgColor};color:${color};border:1px solid ${bdColor}">${label}</span>
          ${aiTag}
        </div>
      </div>
      <button class="day-event-del-btn" data-id="${ev.id}" data-date="${ev.date}" title="Delete event" aria-label="Delete event">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    </div>`;
  }).join("");

  list.querySelectorAll(".day-event-del-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteCalendarEvent(btn.dataset.id, btn.dataset.date);
    });
  });

  list.querySelectorAll(".day-event-title").forEach(el => {
    const openEdit = () => {
      const ev = (calendarEvents[el.dataset.date] || []).find(e => e.id === el.dataset.id);
      if (ev) openEventModal(ev);
    };
    el.addEventListener("click", openEdit);
    el.addEventListener("keypress", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openEdit(); } });
  });
}

// Add a manual calendar event for the selected date
async function addCalendarEventManual(dateStr, title, type) {
  try {
    const res = await fetch(`${API_BASE}/calendar/events`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, date: dateStr, type, source: "manual" })
    });
    if (!res.ok) return;
    const data = await res.json();
    if (data.success && data.event) {
      if (!calendarEvents[dateStr]) calendarEvents[dateStr] = [];
      calendarEvents[dateStr].push(data.event);
      renderCalendar();
      renderDayEvents(dateStr);
      updateDayEventsCount(dateStr);
    }
  } catch (e) {
    console.warn("[OrbixAI] addCalendarEventManual error:", e);
  }
}

// Delete a calendar event
async function deleteCalendarEvent(eventId, dateStr) {
  try {
    const res = await fetch(`${API_BASE}/calendar/events/${eventId}`, { method: "DELETE" });
    if (!res.ok) return;
    if (calendarEvents[dateStr]) {
      calendarEvents[dateStr] = calendarEvents[dateStr].filter(e => e.id !== eventId);
      if (calendarEvents[dateStr].length === 0) delete calendarEvents[dateStr];
    }
    renderCalendar();
    if (selectedCalDate === dateStr) {
      renderDayEvents(dateStr);
      updateDayEventsCount(dateStr);
    }
  } catch (e) {
    console.warn("[OrbixAI] deleteCalendarEvent error:", e);
  }
}

// Update an existing calendar event via backend
async function updateCalendarEvent(eventId, fields, origDate) {
  try {
    const res = await fetch(`${API_BASE}/calendar/events/${eventId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(fields)
    });
    if (!res.ok) return;
    const data = await res.json();
    if (data.success && data.event) {
      const ev = data.event;
      // Remove from old date slot
      if (calendarEvents[origDate]) {
        calendarEvents[origDate] = calendarEvents[origDate].filter(e => e.id !== eventId);
        if (calendarEvents[origDate].length === 0) delete calendarEvents[origDate];
      }
      // Insert into new date slot
      if (!calendarEvents[ev.date]) calendarEvents[ev.date] = [];
      if (!calendarEvents[ev.date].find(e => e.id === ev.id)) calendarEvents[ev.date].push(ev);

      renderCalendar();
      if (selectedCalDate === origDate || selectedCalDate === ev.date) {
        selectedCalDate = ev.date;
        renderDayEvents(ev.date);
        updateDayEventsCount(ev.date);
      }
    }
  } catch (e) {
    console.warn("[OrbixAI] updateCalendarEvent error:", e);
  }
}

// ── Event Edit Modal ─────────────────────────────────────────────────────────
function openEventModal(ev) {
  document.getElementById("event-modal-id").value         = ev.id;
  document.getElementById("event-modal-orig-date").value  = ev.date;
  document.getElementById("event-modal-title-input").value = ev.title || "";
  document.getElementById("event-modal-date").value       = ev.date || "";
  document.getElementById("event-modal-time").value       = ev.time || "";
  document.getElementById("event-modal-type").value       = ev.type || "general";
  document.getElementById("event-modal-desc").value       = ev.description || "";

  const modal   = document.getElementById("event-modal");
  const overlay = document.getElementById("event-modal-overlay");
  if (modal)   modal.classList.add("active");
  if (overlay) overlay.classList.add("active");
  document.body.style.overflow = "hidden";
}

function closeEventModal() {
  const modal   = document.getElementById("event-modal");
  const overlay = document.getElementById("event-modal-overlay");
  if (modal)   modal.classList.remove("active");
  if (overlay) overlay.classList.remove("active");
  document.body.style.overflow = "";
}

function initEventModal() {
  const closeBtn   = document.getElementById("event-modal-close");
  const overlay    = document.getElementById("event-modal-overlay");
  const deleteBtn  = document.getElementById("event-modal-delete");
  const form       = document.getElementById("event-form");

  if (closeBtn) closeBtn.addEventListener("click", closeEventModal);
  if (overlay)  overlay.addEventListener("click", closeEventModal);

  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const eventId   = document.getElementById("event-modal-id").value;
      const origDate  = document.getElementById("event-modal-orig-date").value;
      const fields    = {
        title:       document.getElementById("event-modal-title-input").value.trim(),
        date:        document.getElementById("event-modal-date").value,
        time:        document.getElementById("event-modal-time").value,
        type:        document.getElementById("event-modal-type").value,
        description: document.getElementById("event-modal-desc").value.trim(),
      };
      if (!fields.title || !fields.date) return;
      closeEventModal();
      await updateCalendarEvent(eventId, fields, origDate);
    });
  }

  if (deleteBtn) {
    deleteBtn.addEventListener("click", async () => {
      const eventId  = document.getElementById("event-modal-id").value;
      const origDate = document.getElementById("event-modal-orig-date").value;
      closeEventModal();
      await deleteCalendarEvent(eventId, origDate);
    });
  }
}

// ── Calendar init ─────────────────────────────────────────────────────────────
function initCalendar() {
  renderCalendar();
  const prev = document.getElementById("cal-prev");
  const next = document.getElementById("cal-next");
  if (prev) prev.addEventListener("click", () => {
    calDate.setMonth(calDate.getMonth() - 1);
    fetchCalendarEvents(calDate.getFullYear(), calDate.getMonth() + 1);
  });
  if (next) next.addEventListener("click", () => {
    calDate.setMonth(calDate.getMonth() + 1);
    fetchCalendarEvents(calDate.getFullYear(), calDate.getMonth() + 1);
  });

  // Day events panel close
  const closePanel = document.getElementById("day-events-close");
  if (closePanel) closePanel.addEventListener("click", hideDayPanel);

  // Day event inline-add
  const addBtn   = document.getElementById("day-event-add-btn");
  const addInput = document.getElementById("day-event-input");
  function handleAddEvent() {
    if (!addInput || !selectedCalDate) return;
    const title = addInput.value.trim();
    if (!title) return;
    const type = document.getElementById("day-event-type")?.value || "general";
    addInput.value = "";
    addCalendarEventManual(selectedCalDate, title, type);
  }
  if (addBtn)   addBtn.addEventListener("click", handleAddEvent);
  if (addInput) addInput.addEventListener("keypress", (e) => { if (e.key === "Enter") handleAddEvent(); });

  // Load events for current month
  fetchCalendarEvents(calDate.getFullYear(), calDate.getMonth() + 1);
}

function initTodayInfo() {
  const now    = new Date();
  const dateEl = document.getElementById("today-date");
  const dayEl  = document.getElementById("today-day");
  if (dateEl) dateEl.textContent = now.getDate();
  if (dayEl)  dayEl.textContent  = now.toLocaleString("default", { weekday: "long", month: "short", year: "numeric" });
}

// ── TO-DO LIST ────────────────────────────────────────────────────────────────
let todos = [];

// Migrate legacy todos (may lack source field)
function loadTodos() {
  try {
    const raw = JSON.parse(localStorage.getItem("orbix-todos") || "[]");
    todos = raw.map(t => ({
      text:      t.text || "",
      done:      !!t.done,
      source:    t.source || "manual",
      createdAt: t.createdAt || 0,
    }));
  } catch (_) {
    todos = [];
  }
}

function saveTodos() {
  localStorage.setItem("orbix-todos", JSON.stringify(todos));
}

function renderTodos() {
  const list    = document.getElementById("todo-list");
  const countEl = document.getElementById("todo-count");
  if (!list) return;

  const pending = todos.filter(t => !t.done).length;
  if (countEl) countEl.textContent = pending;

  if (todos.length === 0) {
    list.innerHTML = `<div class="todo-empty">No tasks yet</div>`;
    return;
  }

  list.innerHTML = todos.map((t, i) => {
    let badge = "";
    if (t.source === "ai_travel")  badge = `<span class="todo-source-badge travel">Travel</span>`;
    else if (t.source === "ai_meeting") badge = `<span class="todo-source-badge meeting">Meeting</span>`;

    return `<div class="todo-item" role="listitem">
      <div class="todo-check ${t.done ? "done" : ""}" data-idx="${i}" role="checkbox" aria-checked="${t.done}" tabindex="0" aria-label="Toggle task completion"></div>
      <span class="todo-text ${t.done ? "done" : ""}">${escapeHtml(t.text)}</span>
      ${badge}
      <button class="todo-del-btn" data-idx="${i}" aria-label="Delete task">✕</button>
    </div>`;
  }).join("");

  list.querySelectorAll(".todo-check").forEach(el => {
    const toggle = () => {
      todos[+el.dataset.idx].done = !todos[+el.dataset.idx].done;
      saveTodos();
      renderTodos();
    };
    el.addEventListener("click", toggle);
    el.addEventListener("keypress", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
  });
  list.querySelectorAll(".todo-del-btn").forEach(el => {
    el.addEventListener("click", () => {
      todos.splice(+el.dataset.idx, 1);
      saveTodos();
      renderTodos();
    });
  });
}

// Auto-add AI-generated todo items (called from stream response handler)
function autoAddTodos(items) {
  (items || []).forEach(item => {
    if (!item.text) return;
    todos.unshift({ text: item.text, done: false, source: item.source || "manual", createdAt: Date.now() });
  });
  saveTodos();
  renderTodos();
}

function initTodos() {
  loadTodos();
  renderTodos();
  const input  = document.getElementById("todo-input");
  const addBtn = document.getElementById("todo-add-btn");

  function addTodo() {
    if (!input) return;
    const text = input.value.trim();
    if (!text) return;
    todos.unshift({ text, done: false, source: "manual", createdAt: Date.now() });
    saveTodos();
    renderTodos();
    input.value = "";
    input.focus();
  }
  if (addBtn) addBtn.addEventListener("click", addTodo);
  if (input)  input.addEventListener("keypress", e => { if (e.key === "Enter") addTodo(); });
}

// ── Personal Assistant: greeting + welcome hero ────────────────────────────────
let ASSISTANT_NAME = "";  // filled from the logged-in Google account on load

function greetingPart() {
  const h = new Date().getHours();
  if (h < 12) return "Good morning";
  if (h < 17) return "Good afternoon";
  if (h < 21) return "Good evening";
  return "Working late";
}

function heroGreeting() {
  return ASSISTANT_NAME ? `${greetingPart()}, ${ASSISTANT_NAME}` : greetingPart();
}

function setGreeting() {
  const line = document.getElementById("greeting-line");
  const sub  = document.getElementById("greeting-sub");
  const namePart = ASSISTANT_NAME ? `, <span class="greeting-accent">${escapeHtml(ASSISTANT_NAME)}</span>` : "";
  if (line) line.innerHTML = `${greetingPart()}${namePart}`;
  if (sub)  sub.textContent = "How can I help you today?";
}

// Fetch the logged-in user's name from the backend (Gmail account) and apply it
async function loadUserName() {
  try {
    const res  = await fetch(`${API_BASE}/auth/profile`);
    const data = await res.json();
    if (data && data.name) {
      ASSISTANT_NAME = String(data.name).trim().split(/\s+/)[0] || String(data.name).trim();
      setGreeting();
      const heroTitle = document.querySelector("#welcome-hero .welcome-title");
      if (heroTitle) heroTitle.textContent = heroGreeting();
    }
  } catch (_) { /* not authenticated yet — greeting stays generic */ }
}

// Clickable suggestion chips fill the input and send
window.orbixSuggest = function (text) {
  if (userInput) { userInput.value = text; userInput.focus(); }
  sendMessage();
};

const SUGGESTIONS = [
  { label: "Schedule a meeting",  prompt: "Schedule a Google Meet with example@gmail.com to discuss the project roadmap",
    icon: '<path d="M8 2v4M16 2v4M3 10h18M5 4h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z"/>' },
  { label: "Check my emails",     prompt: "Show me my latest emails",
    icon: '<rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-10 5L2 7"/>' },
  { label: "Plan a trip",         prompt: "Plan a 3-day trip to Tokyo",
    icon: '<path d="M17.8 19.2 16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.2c.4-.3.6-.7.5-1.2z"/>' },
  { label: "Summarize my day",    prompt: "Give me a quick summary of my schedule and tasks today",
    icon: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/>' },
];

function showWelcomeHero() {
  if (!chatContainer) return;
  const hero = document.createElement("div");
  hero.className = "welcome-hero";
  hero.id = "welcome-hero";

  const chips = SUGGESTIONS.map(s => {
    const safePrompt = s.prompt.replace(/'/g, "\\'").replace(/"/g, "&quot;");
    return `<button class="welcome-chip" type="button" onclick="orbixSuggest('${safePrompt}')">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${s.icon}</svg>
      ${escapeHtml(s.label)}
    </button>`;
  }).join("");

  hero.innerHTML = `
    <div class="welcome-orb">O</div>
    <div class="welcome-title">${escapeHtml(heroGreeting())}</div>
    <div class="welcome-sub">I can manage your calendar, draft and send emails, set up Google Meets, plan trips, and more. Try one of these to get started.</div>
    <div class="welcome-chips">${chips}</div>`;
  chatContainer.appendChild(hero);
}

// ── Today agenda (driven by calendar events) ────────────────────────────────────
function renderTodayAgenda() {
  const el = document.getElementById("today-agenda");
  if (!el) return;
  const now = new Date();
  const ds = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
  const evs = (calendarEvents[ds] || []).slice()
    .sort((a, b) => (a.time || "~").localeCompare(b.time || "~"));

  if (evs.length === 0) {
    el.innerHTML = `<div class="today-agenda-empty">No events scheduled today.</div>`;
    return;
  }

  el.innerHTML = evs.slice(0, 4).map(ev => {
    const color = ev.color || EVENT_TYPE_COLORS[ev.type] || "#8b5cf6";
    const time  = ev.time ? escapeHtml(ev.time) : "All day";
    return `<div class="agenda-item">
      <div class="agenda-bar" style="background:${color}"></div>
      <div class="agenda-body">
        <span class="agenda-title">${escapeHtml(ev.title)}</span>
        <span class="agenda-time">${time}</span>
      </div>
    </div>`;
  }).join("") + (evs.length > 4 ? `<div class="today-agenda-empty">+${evs.length - 4} more…</div>` : "");
}

// ── App Init ──────────────────────────────────────────────────────────────────
window.addEventListener("load", () => {
  initializeTheme();
  setGreeting();
  loadUserName();
  if (userInput) userInput.focus();
  showWelcomeHero();
  startEmailAutoRefresh();
  startHealthCheck();
  initCalendar();
  initTodayInfo();
  initTodos();
  initEventModal();
});

window.addEventListener("beforeunload", () => {
  stopEmailAutoRefresh();
  stopHealthCheck();
});
