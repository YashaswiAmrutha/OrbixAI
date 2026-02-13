const { app, BrowserWindow, globalShortcut } = require("electron");
const path = require("path");

let win;

function createWindow() {
  win = new BrowserWindow({
    width: 420,
    height: 720,
    resizable: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js")
    }
  });

  // Load your EXISTING frontend
  win.loadFile(path.join(__dirname, "../frontend/index.html"));
}

app.whenReady().then(() => {
  createWindow();

  // GLOBAL SHORTCUT (Ctrl + Space)
  globalShortcut.register("CommandOrControl+Space", () => {
    win.show();
    win.focus();
    win.webContents.executeJavaScript(`
      document.getElementById("mic-btn")?.click();
    `);
  });
});

app.on("will-quit", () => {
  globalShortcut.unregisterAll();
});
