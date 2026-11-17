// The page's only bridge to the desktop: these calls and nothing else.
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("streamLog", {
  once: (args) => ipcRenderer.invoke("cli:once", args),
  start: (args) => ipcRenderer.invoke("job:start", args),
  cancel: () => ipcRenderer.invoke("job:cancel"),
  onEvent: (callback) => ipcRenderer.on("job:event", (_event, data) => callback(data)),
  settings: () => ipcRenderer.invoke("settings:get"),
  saveSettings: (changes) => ipcRenderer.invoke("settings:set", changes),
  chooseVideo: (folder) => ipcRenderer.invoke("dialog:video", folder),
  chooseFolder: (folder) => ipcRenderer.invoke("dialog:folder", folder),
  exists: (file) => ipcRenderer.invoke("files:exists", file),
  videos: (folder) => ipcRenderer.invoke("files:videos", folder),
  open: (file) => ipcRenderer.invoke("shell:open", file),
  show: (file) => ipcRenderer.invoke("shell:show", file),
  openLog: () => ipcRenderer.invoke("shell:log"),
});
