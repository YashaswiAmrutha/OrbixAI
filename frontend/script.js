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
    voiceMuteBtn.classList.add("muted");
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
  voiceMuteBtn.classList.toggle("muted");
}

themeToggleBtn.addEventListener("click", toggleTheme);
voiceMuteBtn.addEventListener("click", toggleVoiceMute);

sendBtn.addEventListener("click", sendMessage);

userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

micBtn.addEventListener("click", startVoice);

if (mailRefreshBtn) mailRefreshBtn.addEventListener("click", () => fetchEmails(true));
if (mailToggleBtn) mailToggleBtn.addEventListener("click", toggleMailWidget);
if (modalClose) modalClose.addEventListener("click", closeMailModal);
if (modalCancel) modalCancel.addEventListener("click", closeMailModal);
if (modalOverlay) modalOverlay.addEventListener("click", closeMailModal);
if (mailForm) mailForm.addEventListener("submit", handleMailSubmit);

function scrollToBottom() {
  setTimeout(() => {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }, 50);
}

function addMessage(text, role) {
  const wrapper = document.createElement("div");
  wrapper.className = `message-wrapper ${role}`;

  const messageDiv = document.createElement("div");
  messageDiv.className = "message";
  messageDiv.textContent = text;

  wrapper.appendChild(messageDiv);
  chatContainer.appendChild(wrapper);
  scrollToBottom();
}

function addStatusMessage(text) {
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

async function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return;

  addMessage(message, "user");
  userInput.value = "";
  sendBtn.disabled = true;

  const statusEl = addStatusMessage("Classifying intent\u2026");

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    });

    const data = await res.json();

    removeElement(statusEl);

    if (data.intent) {
      const intentDisplay = data.intent.replace(/_/g, " ");
      addStatusMessage(`Classified intent: ${intentDisplay}`);
    }

    if (data.action === "open_mail_modal") {
      openMailModal(data.parameters || {});
    } else if (data.reply) {
      addMessage(data.reply, "assistant");
      speak(data.reply);
    }
  } catch (error) {
    removeElement(statusEl);
    addMessage("Connection error. Please check if the backend is running.", "assistant");
  } finally {
    sendBtn.disabled = false;
    userInput.focus();
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

      const statusEl = addStatusMessage("Processing voice\u2026");

      try {
        const response = await fetch("/voice", {
          method: "POST",
          body: formData
        });

        const data = await response.json();

        removeElement(statusEl);

        if (data.intent) {
          const intentDisplay = data.intent.replace(/_/g, " ");
          addStatusMessage(`Classified intent: ${intentDisplay}`);
        }

        if (data.text) {
          addMessage(data.text, "user");
        }
        if (data.reply) {
          addMessage(data.reply, "assistant");
          speak(data.reply);
        }
      } catch (error) {
        removeElement(statusEl);
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
    const response = await fetch("/emails/latest?max_results=10", {
      signal: controller.signal
    });

    clearTimeout(timeout);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
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
    if (error.name === "AbortError") {
      displayEmailError("Connection timeout. Retrying…");
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
  let friendlyMsg = "Unable to load emails";
  if (errorMsg.includes("not initialized")) {
    friendlyMsg = "Gmail not connected. Check credentials.";
  } else if (errorMsg.includes("timed out") || errorMsg.includes("timeout")) {
    friendlyMsg = "Gmail timed out. Will retry…";
  } else if (errorMsg.includes("server")) {
    friendlyMsg = "Server unreachable";
  }
  mailList.innerHTML = `<div class="mail-item placeholder error-state">
    <p>${friendlyMsg}</p>
    <button class="mail-retry-btn" onclick="fetchEmails(true)">Retry</button>
  </div>`;
}

function openMailModal(prefill) {
  if (prefill && prefill.recipient_email) {
    const recipientInput = document.getElementById("recipient-email");
    if (recipientInput) recipientInput.value = prefill.recipient_email;
  }
  mailModal.classList.add("active");
  modalOverlay.classList.add("active");
  document.body.style.overflow = "hidden";
}

function closeMailModal() {
  mailModal.classList.remove("active");
  modalOverlay.classList.remove("active");
  document.body.style.overflow = "";
  mailForm.reset();
}

function toggleMailWidget() {
  mailContainer.classList.toggle("collapsed");
}

async function handleMailSubmit(e) {
  e.preventDefault();

  const toEmail = document.getElementById("recipient-email").value;
  const mailPrompt = document.getElementById("mail-prompt").value;
  const createMeet = document.getElementById("create-meet").checked;
  const useLLM = document.getElementById("use-llm").checked;

  if (!toEmail || !mailPrompt) {
    addMessage("Please fill in all fields", "assistant");
    return;
  }

  closeMailModal();

  try {
    let meetLink = null;

    // Step 1: Create Google Meet if requested
    if (createMeet) {
      addStatusMessage("Creating Google Meet…");

      const meetRes = await fetch("/meetings/create", {
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
    addStatusMessage("Sending email…");

    const emailPayload = {
      to_email: toEmail,
      use_llm: useLLM,
      user_prompt: mailPrompt,
      recipient_name: toEmail.split("@")[0]
    };

    if (meetLink) {
      emailPayload.meeting_link = meetLink;
    }

    if (!useLLM) {
      emailPayload.subject = createMeet ? "Meeting Invitation" : "Message from OrbixAI";
      emailPayload.body = mailPrompt + (meetLink ? `\n\nGoogle Meet Link: ${meetLink}` : "");
    }

    const emailRes = await fetch("/emails/send", {
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
    const response = await fetch("/health", { signal: controller.signal });
    clearTimeout(timeout);

    if (response.ok) {
      setOnlineStatus(true);
    } else {
      setOnlineStatus(false);
    }
  } catch (error) {
    clearTimeout(timeout);
    setOnlineStatus(false);
  }
}

function setOnlineStatus(isOnline) {
  if (!headerStatus || !statusText) return;

  if (isOnline) {
    headerStatus.classList.remove("offline");
    statusText.textContent = "Online";
  } else {
    headerStatus.classList.add("offline");
    statusText.textContent = "Offline";
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

window.addEventListener("load", () => {
  initializeTheme();
  userInput.focus();
  addStatusMessage("Welcome to OrbixAI. Start chatting or use voice.");
  startEmailAutoRefresh();
  startHealthCheck();
});

window.addEventListener("beforeunload", () => {
  stopEmailAutoRefresh();
  stopHealthCheck();
});

