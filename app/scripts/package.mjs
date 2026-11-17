// Build "Stream Log Setup <version>.exe" (release/): an NSIS installer from electron-builder.
//
// Inside: the window (electron/, ui/) and resources/backend: python.org's embeddable Python
// (Authenticode-signed by the PSF, checked below) with the backend's locked dependencies
// (backend/uv.lock) plus the stream_log package, and FFmpeg from this machine's PATH.
// The CUDA libraries (~2 GB) stay out: the app downloads them on first run
// with llama.cpp and the models (Settings > first-time setup, `stream_log setup`).
//
//   npm run installer

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const APP = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ROOT = path.dirname(APP);
const BACKEND = path.join(ROOT, "backend");
const RELEASE = path.join(APP, "release");
const CACHE = path.join(RELEASE, "cache");
const STAGE = path.join(RELEASE, "stage");
const PYTHON = "3.12.10"; // last 3.12 with Windows binaries; the backend venv is 3.12
const PYTHON_ZIP = `https://www.python.org/ftp/python/${PYTHON}/python-${PYTHON}-embed-amd64.zip`;
const VC_RUNTIME = ["msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "concrt140.dll",
                    "vcomp140.dll"];
const SYSTEM32 = path.join(process.env.SystemRoot ?? "C:\\Windows", "System32");
const TAR = path.join(SYSTEM32, "tar.exe");
const UV = process.env.UV ?? path.join(process.env.LOCALAPPDATA ?? "", "Microsoft", "WinGet",
  "Packages", "astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe", "uv.exe");

const version = JSON.parse(fs.readFileSync(path.join(APP, "package.json"), "utf8")).version;
const step = (text) => console.log(`\n== ${text}`);
const run = (exe, list, options = {}) => execFileSync(exe, list, { stdio: "inherit", ...options });
const output = (exe, list) => execFileSync(exe, list, { encoding: "utf8" }).trim();
const noCache = (src) => !src.includes("__pycache__");

function signer(file) {
  return output("powershell.exe", ["-NoProfile", "-Command",
    `$s = Get-AuthenticodeSignature -LiteralPath '${file}'; "$($s.Status)|$($s.SignerCertificate.Subject)"`]);
}

async function fetchTo(url, dest) {
  if (fs.existsSync(dest)) return;
  console.log(`downloading ${url}`);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.writeFileSync(`${dest}.part`, Buffer.from(await response.arrayBuffer()));
  fs.renameSync(`${dest}.part`, dest);
}

const megabytes = (dir) => {
  let total = 0;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true, recursive: true })) {
    if (entry.isFile()) total += fs.statSync(path.join(entry.parentPath, entry.name)).size;
  }
  return Math.round(total / 1e6);
};

// ---------------------------------------------------------------------- the window

function writeApp(appDir) {
  fs.mkdirSync(path.join(appDir, "electron"), { recursive: true });
  for (const name of ["main.cjs", "preload.cjs", "icon.ico"]) {
    fs.copyFileSync(path.join(APP, "electron", name), path.join(appDir, "electron", name));
  }
  fs.cpSync(path.join(APP, "ui"), path.join(appDir, "ui"), { recursive: true });
  fs.writeFileSync(path.join(appDir, "package.json"), JSON.stringify({
    name: "stream-log", productName: "Stream Log", version, main: "electron/main.cjs",
    description: "Stream Log", author: "remispi", license: "MIT",
  }, null, 1));
}

// ---------------------------------------------------------------------- Python + backend

async function writeBackend(res) {
  const py = path.join(res, "backend", "python");
  const site = path.join(py, "Lib", "site-packages");

  step(`Python ${PYTHON} (embeddable, python.org)`);
  const pyZip = path.join(CACHE, path.basename(PYTHON_ZIP));
  await fetchTo(PYTHON_ZIP, pyZip);
  fs.mkdirSync(py, { recursive: true });
  run(TAR, ["-xf", pyZip, "-C", py]);
  const signed = signer(path.join(py, "python.exe"));
  if (!signed.startsWith("Valid|") || !signed.includes("Python Software Foundation")) {
    throw new Error(`python.exe is not signed by the PSF: ${signed}`);
  }
  console.log(`python.exe signature: ${signed}`);
  const tag = PYTHON.split(".").slice(0, 2).join("");
  // The embeddable build ignores PYTHONPATH and reads sys.path from this file only.
  fs.writeFileSync(path.join(py, `python${tag}._pth`),
    `python${tag}.zip\r\n.\r\nLib\\site-packages\r\nimport site\r\n`);
  for (const name of VC_RUNTIME) {
    const system = path.join(SYSTEM32, name);
    if (!fs.existsSync(system)) {
      throw new Error(`${name} isn't in ${SYSTEM32}: install the Microsoft Visual C++ ` +
                      "Redistributable (x64) on this machine, then build again");
    }
    fs.copyFileSync(system, path.join(py, name));
  }

  step("backend dependencies (backend/uv.lock, CUDA wheels left out)");
  const requirements = path.join(CACHE, "requirements.txt");
  fs.mkdirSync(CACHE, { recursive: true });
  run(UV, ["export", "--project", BACKEND, "--frozen", "--no-dev", "--no-emit-project",
           "--no-hashes", "--format", "requirements-txt",
           "-o", requirements]);
  const lines = fs.readFileSync(requirements, "utf8").split(/\r?\n/);
  fs.writeFileSync(requirements, lines.filter((l) => !/^\s*(nvidia-|-e\s)/.test(l)).join("\n"));
  run(UV, ["pip", "install", "--target", site, "--python", path.join(py, "python.exe"),
           "--no-deps", "--link-mode", "copy", "--compile-bytecode", "-r", requirements]);
  fs.cpSync(path.join(BACKEND, "src", "stream_log"), path.join(site, "stream_log"),
            { recursive: true, filter: noCache });
  // First run downloads the CUDA wheels: url, hash and size from the lock, read by the backend's
  // own parser so both agree.
  const wheels = output(path.join(py, "python.exe"), ["-c",
    "import json, sys, dataclasses; from stream_log.engine.first_run import wheels_from_lock; " +
    "print(json.dumps([dataclasses.asdict(w) for w in wheels_from_lock(" +
    "open(sys.argv[1], encoding='utf-8').read())]))", path.join(BACKEND, "uv.lock")]);
  if (JSON.parse(wheels).length === 0) throw new Error("no nvidia wheels found in uv.lock");
  fs.writeFileSync(path.join(site, "stream_log", "engine", "gpu_wheels.json"), wheels);
  run(path.join(py, "python.exe"), ["-m", "compileall", "-q", path.join(site, "stream_log")]);
  // The bundled command line must start: imports and the status check on this machine.
  run(path.join(py, "python.exe"), ["-s", "-c", "import stream_log.cli, stream_log.run, " +
                                    "stream_log.games, stream_log.engine.pipeline; print('imports ok')"]);

  step("FFmpeg");
  const ffmpeg = output("where.exe", ["ffmpeg"]).split(/\r?\n/)[0];
  const ffDir = path.join(res, "backend", "ffmpeg");
  fs.mkdirSync(ffDir, { recursive: true });
  for (const name of ["ffmpeg.exe", "ffprobe.exe"]) {
    fs.copyFileSync(path.join(path.dirname(ffmpeg), name), path.join(ffDir, name));
    console.log(output(path.join(ffDir, name), ["-hide_banner", "-version"]).split(/\r?\n/)[0]);
  }
  const license = path.join(path.dirname(path.dirname(ffmpeg)), "LICENSE");
  if (fs.existsSync(license)) fs.copyFileSync(license, path.join(ffDir, "LICENSE.txt"));
}

// ---------------------------------------------------------------------- build

step(`staging -> ${STAGE}`);
fs.rmSync(STAGE, { recursive: true, force: true });
writeApp(path.join(STAGE, "app"));
await writeBackend(path.join(STAGE, "resources"));
console.log(`\nbackend: ${megabytes(path.join(STAGE, "resources"))} MB`);

step("installer (electron-builder, NSIS)");
const electronVersion = JSON.parse(fs.readFileSync(
  path.join(APP, "node_modules", "electron", "package.json"), "utf8")).version;
// The window needs no node_modules; a beforeBuild hook returning false stops electron-builder
// from installing or copying the project's dev dependencies into the app.
const noNodeModules = path.join(STAGE, "before-build.cjs");
fs.writeFileSync(noNodeModules, "module.exports = () => false;\n");
const config = {
  appId: "StreamLog", // = app.setAppUserModelId in main.cjs
  productName: "Stream Log",
  copyright: "Copyright (c) 2026 remispi",
  electronVersion,
  directories: { app: path.join(STAGE, "app"), output: RELEASE },
  asar: false,
  beforeBuild: noNodeModules,
  extraResources: [{ from: path.join(STAGE, "resources", "backend"), to: "backend" }],
  win: {
    target: [{ target: "nsis", arch: ["x64"] }],
    icon: path.join(APP, "electron", "icon.ico"),
    signAndEditExecutable: true, // unsigned: no certificate
  },
  nsis: {
    oneClick: false,
    perMachine: false, // per user, no admin prompt
    allowElevation: true,
    allowToChangeInstallationDirectory: true,
    createDesktopShortcut: true,
    createStartMenuShortcut: true,
    shortcutName: "Stream Log",
    installerIcon: path.join(APP, "electron", "icon.ico"),
    uninstallerIcon: path.join(APP, "electron", "icon.ico"),
    deleteAppDataOnUninstall: false, // settings; models and projects live in the data folder
    artifactName: "Stream Log Setup ${version}.exe",
  },
};
const configFile = path.join(STAGE, "electron-builder.json");
fs.writeFileSync(configFile, JSON.stringify(config, null, 1));
run(process.env.ComSpec ?? "cmd.exe",
    ["/d", "/c", "npx", "electron-builder", "--win", "--x64", "--config", configFile,
     "--projectDir", APP, "--publish", "never"], { cwd: APP });
const setup = path.join(RELEASE, `Stream Log Setup ${version}.exe`);
console.log(`\n${setup}: ${Math.round(fs.statSync(setup).size / 1e6)} MB`);
fs.rmSync(STAGE, { recursive: true, force: true });
