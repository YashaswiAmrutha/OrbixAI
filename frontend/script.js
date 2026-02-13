async function sendMessage() {
  const input = document.getElementById("user-input");
  const message = input.value.trim();
  if (!message) return;

  addMessage(message, "user");
  input.value = "";

  const res = await fetch("http://127.0.0.1:8001/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message })
  });

  const data = await res.json();
  if (data.text) {
    addMessage(data.text, "user");
  }
  addMessage(data.reply, "Orbix");
  speak(data.reply);
  
}

let mediaRecorder;
let audioChunks = [];

async function startVoice() {
  audioChunks = [];

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);

  mediaRecorder.start();
  console.log("🎤 Recording started");

  mediaRecorder.ondataavailable = event => {
    audioChunks.push(event.data);
  };

  mediaRecorder.onstop = async () => {
    console.log("🛑 Recording stopped");

    const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
    const formData = new FormData();
    formData.append("file", audioBlob, "voice.wav");

    addMessage("🎤 Processing voice...", "orbii");

    const response = await fetch("http://127.0.0.1:8001/voice", {
      method: "POST",
      body: formData
    });

    const data = await response.json();
    if (data.text) {
      addMessage(data.text, "user");
    }
    addMessage(data.reply, "Orbix");
    speak(data.reply);
    
  };

  // auto-stop after 4 seconds
  setTimeout(() => {
    mediaRecorder.stop();
  }, 4000);
}

function speak(text) {
  const utterance = new SpeechSynthesisUtterance(text);
  speechSynthesis.speak(utterance);
}

function addMessage(text, role) {
  const chat = document.getElementById("chat-container");
  const div = document.createElement("div");
  div.className = role;
  div.innerText = text;
  chat.appendChild(div);
}
