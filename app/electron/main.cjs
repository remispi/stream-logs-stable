// Stream Log's window. The page gets no Node access (contextIsolation on, nodeIntegration off,
// sandboxed); preload.cjs hands it a few actions, and every real job runs as the Python command
// line (`python -m stream_log ...`), whose stdout is one JSON object per line.

const { app, BrowserWindow, dialog, ipcMain, shell } = require("electron");
const { spawn, execFile } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const readline = require("node:readline");

app.setAppUserModelId("StreamLog");
const ICON = path.join(__dirname, "icon.ico");
const SETTINGS = () => path.join(app.getPath("userData"), "settings.json");
const LOG = () => path.join(app.getPath("userData"), "backend.log");

function backend() {
  if (app.isPackaged) {
    const root = path.join(process.resourcesPath, "backend");
    return { python: path.join(root, "python", "python.exe"), ffmpeg: path.join(root, "ffmpeg") };
  }
  const python = process.env.STREAM_LOG_PYTHON
    ?? path.join(__dirname, "..", "..", "backend", ".venv", "Scripts", "python.exe");
  return { python, ffmpeg: null };
}

// ---------------------------------------------------------------------- settings

// Models, analysis projects and caches (the backend's STREAM_LOG_HOME). Large: put it on a big drive.
function defaultDataHome() {
  if (process.env.STREAM_LOG_HOME) return process.env.STREAM_LOG_HOME;
  return path.join(process.env.LOCALAPPDATA ?? app.getPath("userData"), "StreamLog");
}

function defaults() {
  return {
    account: "",
    rawFolder: path.join(app.getPath("videos"), "Raw Streams"),
    dataHome: defaultDataHome(),
    mode: "", // the hardware tier's default
    perHour: 14,
    profile: "",
    findGames: true, // look for TFT games in the video

  };
}

function readSettings() {
  try {
    return { ...defaults(), ...JSON.parse(fs.readFileSync(SETTINGS(), "utf8")) };
  } catch {
    return defaults();
  }
}

function writeSettings(changes) {
  const next = { ...readSettings(), ...changes };
  fs.mkdirSync(path.dirname(SETTINGS()), { recursive: true });
  fs.writeFileSync(SETTINGS(), JSON.stringify(next, null, 1));
  return next;
}

// ---------------------------------------------------------------------- the command line

function childEnv() {
  const settings = readSettings();
  const { ffmpeg } = backend();
  const env = { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONUTF8: "1",
                STREAM_LOG_HOME: settings.dataHome };
  if (ffmpeg) env.PATH = `${ffmpeg};${process.env.PATH ?? ""}`;
  return env;
}

function startCli(args, onEvent) {
  const { python } = backend();
  const child = spawn(python, ["-s", "-m", "stream_log", ...args],
                      { env: childEnv(), windowsHide: true });
  const log = fs.createWriteStream(LOG(), { flags: "a" });
  log.write(`\n=== ${new Date().toISOString()} stream_log ${args.join(" ")}\n`);
  const tail = [];
  child.stderr.on("data", (chunk) => {
    log.write(chunk);
    tail.push(...chunk.toString("utf8").split(/\r?\n/).filter(Boolean));
    tail.splice(0, Math.max(0, tail.length - 40));
  });
  readline.createInterface({ input: child.stdout }).on("line", (line) => {
    try { onEvent(JSON.parse(line)); } catch { log.write(`${line}\n`); }
  });
  const done = new Promise((resolve) => {
    child.on("error", (error) => resolve({ code: -1, error: String(error), tail }));
    child.on("close", (code) => { log.end(); resolve({ code, tail }); });
  });
  return { child, done };
}

function killTree(child) {
  // The backend may have started llama-server: end the whole process tree, not just python.
  if (child && child.exitCode === null) {
    execFile("taskkill", ["/pid", String(child.pid), "/T", "/F"], () => {});
  }
}

let job = null; // the running list, setup or sync job (one at a time)
let win = null;

// One-shot commands (VOD lists, status): resolves with the command's answer.
ipcMain.handle("cli:once", async (_event, args) => {
  let answer = null;
  const { done } = startCli(args, (event) => {
    if (!["step", "log", "steps"].includes(event.type)) answer = event;
  });
  const result = await done;
  if (answer) return answer;
  return { type: "error", text: result.error
           ?? (result.tail.slice(-3).join("\n") || `stopped with exit code ${result.code}`) };
});

ipcMain.handle("job:start", (_event, args) => {
  if (job) return { ok: false, text: "Something is already running." };
  const running = startCli(args, (event) => {
    if (event.type === "done") rememberOutputs(event);
    win?.webContents.send("job:event", event);
  });
  job = running;
  running.done.then((result) => {
    job = null;
    // `cancelled`: stopped from the window (taskkill /F ends it with code 1, like a crash).
    // `error`: Python couldn't be started at all.
    win?.webContents.send("job:event", { type: "exit", code: result.code,
                                         cancelled: Boolean(running.cancelled),
                                         error: result.error, tail: result.tail.slice(-8) });
  });
  return { ok: true };
});

function cancelJob() {
  if (!job) return;
  job.cancelled = true;
  killTree(job.child);
}

ipcMain.handle("job:cancel", () => { cancelJob(); return true; });
ipcMain.handle("settings:get", () => readSettings());
ipcMain.handle("settings:set", (_event, changes) => writeSettings(changes));

ipcMain.handle("dialog:video", async (_event, folder) => {
  const result = await dialog.showOpenDialog(win, {
    title: "Choose a stream video", defaultPath: folder || undefined, properties: ["openFile"],
    filters: [{ name: "Videos", extensions: ["mp4", "mkv", "mov", "ts", "flv", "webm"] }],
  });
  return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle("dialog:folder", async (_event, folder) => {
  const result = await dialog.showOpenDialog(win, {
    defaultPath: folder || undefined, properties: ["openDirectory", "createDirectory"] });
  return result.canceled ? null : result.filePaths[0];
});

// The page can only open or reveal the lists a job wrote (paths the backend reported), and only
// ask about video files: even a compromised page can't launch programs or probe other files.
const VIDEO = /\.(mp4|mkv|mov|ts|flv|webm)$/i;
const LIST = /\.timestamps\.(html|txt)$/i;
const outputs = new Set();
const key = (file) => path.resolve(String(file)).toLowerCase();

function rememberOutputs(event) {
  for (const file of [event.html, event.text]) {
    if (typeof file === "string" && LIST.test(file)) outputs.add(key(file));
  }
}

const isOutput = (file) => typeof file === "string" && outputs.has(key(file));

ipcMain.handle("files:exists", (_event, file) =>
  typeof file === "string" && VIDEO.test(file) && fs.existsSync(file));
ipcMain.handle("files:videos", (_event, folder) => {
  try {
    return fs.readdirSync(String(folder)).filter((name) => VIDEO.test(name));
  } catch {
    return [];
  }
});
ipcMain.handle("shell:open", (_event, file) =>
  isOutput(file) ? shell.openPath(file) : "not a list this app wrote");
ipcMain.handle("shell:show", (_event, file) => {
  if (!isOutput(file)) return false;
  shell.showItemInFolder(file);
  return true;
});
ipcMain.handle("shell:log", () => shell.openPath(LOG()));

// ---------------------------------------------------------------------- window

function createWindow() {
  win = new BrowserWindow({
    width: 980, height: 860, minWidth: 560, minHeight: 600, title: "Stream Log",
    icon: ICON, autoHideMenuBar: true, backgroundColor: "#161513", show: false,
    webPreferences: { preload: path.join(__dirname, "preload.cjs"), contextIsolation: true,
                      nodeIntegration: false, sandbox: true },
  });
  win.once("ready-to-show", () => win.show());
  win.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  win.webContents.on("will-navigate", (event) => event.preventDefault());
  win.on("close", (event) => {
    if (!job) return;
    const choice = dialog.showMessageBoxSync(win, {
      type: "question", buttons: ["Stop it and quit", "Keep going"], defaultId: 1, cancelId: 1,
      message: "A job is still running.",
      detail: "Quitting stops it. Run it again later: finished steps are kept.",
    });
    if (choice === 1) event.preventDefault();
    else cancelJob();
  });
  win.loadFile(path.join(__dirname, "..", "ui", "index.html"));
  screenshotMode();
}

// `electron . --screenshot out.png`: load, optionally run SHOT_JS in the page, capture, quit.
// For checking the window without clicking through it (development only).
function screenshotMode() {
  const at = process.argv.indexOf("--screenshot");
  if (at === -1 || app.isPackaged) return;
  const next = process.argv[at + 1];
  const target = next && !next.startsWith("--") ? path.resolve(next)
    : path.join(app.getPath("userData"), "screenshot.png");
  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  win.webContents.once("did-finish-load", async () => {
    await wait(Number(process.env.SHOT_DELAY ?? 6000));
    if (process.env.SHOT_JS) {
      await win.webContents.executeJavaScript(process.env.SHOT_JS).catch(() => {});
      await wait(Number(process.env.SHOT_AFTER ?? 3000));
    }
    const image = await win.webContents.capturePage();
    fs.writeFileSync(target, image.toPNG());
    app.exit(0);
  });
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => { if (win) { win.restore(); win.focus(); } });
  app.whenReady().then(createWindow);
  app.on("window-all-closed", () => app.quit());
}
