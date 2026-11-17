// Stream Log's page: pick a VOD (from a Twitch account or a file), run the job, show the result.
"use strict";

const api = window.streamLog;
const $ = (id) => document.getElementById(id);

const state = {
  settings: null,
  status: null,
  source: "twitch", // twitch | file
  vods: [],
  vod: null, // the chosen Twitch VOD (twitch tab)
  video: null, // the chosen file (file tab)
  running: null, // "run" | "setup" | "sync"
  steps: [],
  started: 0,
  result: null,
  lastError: "",
};

// ---------------------------------------------------------------------- helpers

function length(seconds) {
  const m = Math.round(seconds / 60);
  return m >= 60 ? `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, "0")}` : `${m} min`;
}

function localDate(iso) {
  const d = new Date(iso);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function shortDate(iso) {
  return new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

/** The stream date in a file name: TwitchStream20261114-… or TwitchDownloader's [11-14-26] …. */
function dateInName(name) {
  let m = name.match(/(20\d{2})(\d{2})(\d{2})/);
  if (m) return `${m[1]}-${m[2]}-${m[3]}`;
  m = name.match(/\[(\d{2})-(\d{2})-(\d{2})\]/);
  if (m) return `20${m[3]}-${m[1]}-${m[2]}`;
  return null;
}

function joinPath(folder, name) {
  return folder.replace(/[\\/]+$/, "") + "\\" + name;
}

function fileName(file) {
  return file.split(/[\\/]/).pop();
}

/** Why `name` can't be a Windows file name, or "" (the backend checks it again: naming.bad_stem). */
function badStem(name) {
  if (!name.trim()) return "Give the file a name.";
  // eslint-disable-next-line no-control-regex
  const bad = name.match(/[<>:"/\\|?*\x00-\x1f]/);
  if (bad) return `A file name can't contain ${/[\x00-\x1f]/.test(bad[0]) ? "control characters" : bad[0]}.`;
  if (name !== name.replace(/[ .]+$/, "")) return "A file name can't end with a space or a dot.";
  if (/^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i.test(name.split(".")[0].trim())) {
    return `"${name}" is reserved by Windows.`;
  }
  if (name.length > 200) return "The file name is too long.";
  return "";
}

function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
}

// ---------------------------------------------------------------------- setup check

async function checkStatus() {
  const args = ["status"];
  if (state.settings.mode) args.push("--mode", state.settings.mode);
  const status = await api.once(args);
  if (status.type !== "status") {
    $("setup").hidden = false;
    $("setup-text").textContent = `Can't check the setup: ${status.text ?? "unknown error"}`;
    $("run-setup").hidden = true;
    return;
  }
  state.status = status;
  $("setup").hidden = status.ready;
  $("run-setup").hidden = false;
  const parts = [];
  if (status.missing.cuda) parts.push("GPU libraries");
  if (status.missing.llama) parts.push("the LLM runtime");
  if (status.missing.models.length) parts.push(`${status.missing.models.length} models`);
  $("setup-text").textContent = `Stream Log needs ${parts.join(", ")} before its first run: ` +
    `about ${status.download_gb} GB into ${status.data_home} (${status.free_disk_gb} GB free).`;
  updateMake();
}

// ---------------------------------------------------------------------- choosing a VOD

function selectTab(source) {
  state.source = source;
  $("tab-twitch").setAttribute("aria-selected", String(source === "twitch"));
  $("tab-file").setAttribute("aria-selected", String(source === "file"));
  $("panel-twitch").hidden = source !== "twitch";
  $("panel-file").hidden = source !== "file";
  updateMake();
}

async function findVods(event) {
  event?.preventDefault();
  const account = $("account").value.trim().replace(/^@/, "");
  if (!account) return;
  $("find").disabled = true;
  $("vods-status").textContent = `Looking for ${account}'s VODs…`;
  $("vods-status").classList.remove("error");
  const answer = await api.once(["vods", account]);
  $("find").disabled = false;
  if (answer.type !== "vods") {
    $("vods-status").textContent = answer.text ?? "Twitch didn't answer.";
    $("vods-status").classList.add("error");
    return;
  }
  state.settings = await api.saveSettings({ account });
  state.vods = answer.vods;
  $("vods-status").textContent = answer.vods.length
    ? `${answer.vods.length} recent VODs (public ones only)` : "No VODs on this account.";
  renderVods();
  fillMatch();
}

function renderVods() {
  const list = $("vods");
  list.replaceChildren();
  for (const vod of state.vods) {
    const row = document.createElement("label");
    row.className = "vod";
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "vod";
    radio.checked = state.vod?.id === vod.id;
    radio.addEventListener("change", () => chooseVod(vod));
    const date = document.createElement("span");
    date.className = "date";
    date.textContent = shortDate(vod.created_at);
    const title = document.createElement("span");
    title.className = "title";
    title.textContent = vod.title;
    title.title = `${vod.title} (${vod.game || "no category"})`;
    const len = document.createElement("span");
    len.className = "len";
    len.textContent = length(vod.length);
    row.append(radio, date, title, len);
    list.append(row);
  }
}

async function chooseVod(vod) {
  state.vod = vod;
  $("save-as").hidden = false;
  $("stem").value = vod.stem;
  await checkAlready();
  updateMake();
}

async function checkAlready() {
  const stem = $("stem").value.trim();
  const exists = !badStem(stem) && await api.exists(joinPath(state.settings.rawFolder, `${stem}.mp4`));
  $("already").hidden = !exists;
  // The same stream saved under another name (e.g. by TwitchDownloader): same date in the name.
  state.sameDay = null;
  if (!exists && state.vod) {
    const day = localDate(state.vod.created_at);
    const names = (await api.videos(state.settings.rawFolder))
      .filter((name) => dateInName(name) === day);
    const channel = (state.vod.channel || "").toLowerCase();
    state.sameDay = names.find((n) => channel && n.toLowerCase().includes(channel)) ?? names[0];
  }
  $("have-it").hidden = !state.sameDay;
  $("have-name").textContent = state.sameDay ?? "";
}

async function useSameDay() {
  if (!state.sameDay) return;
  state.video = joinPath(state.settings.rawFolder, state.sameDay);
  $("video-path").textContent = state.video;
  $("video-path").classList.remove("muted");
  fillMatch();
  $("match").value = state.vod.url;
  selectTab("file");
}

async function chooseVideo() {
  const file = await api.chooseVideo(state.settings.rawFolder);
  if (!file) return;
  state.video = file;
  $("video-path").textContent = file;
  $("video-path").classList.remove("muted");
  if (!state.vods.length && state.settings.account) {
    const answer = await api.once(["vods", state.settings.account]);
    if (answer.type === "vods") state.vods = answer.vods;
  }
  fillMatch();
  updateMake();
}

/** The account's VODs as choices for the file, the one from the file's date first. */
function fillMatch() {
  const select = $("match");
  const day = state.video ? dateInName(fileName(state.video)) : null;
  select.replaceChildren(new Option("None", ""));
  let picked = false;
  for (const vod of state.vods) {
    const option = new Option(`${shortDate(vod.created_at)} · ${vod.title} (${length(vod.length)})`,
                              vod.url);
    if (!picked && day && localDate(vod.created_at) === day) {
      option.selected = true;
      picked = true;
    }
    select.append(option);
  }
}

function updateMake() {
  const ready = state.status?.ready !== false;
  let ok = false;
  let hint = "";
  if (state.source === "twitch") {
    const problem = state.vod ? badStem($("stem").value.trim()) : "";
    ok = Boolean(state.vod) && !problem;
    hint = state.vod ? problem : "Find and pick a VOD first.";
  } else {
    ok = Boolean(state.video);
    hint = state.video ? "" : "Choose a video first.";
  }
  if (!ready) hint = "Finish the first-time setup first.";
  $("make").disabled = !ok || !ready || Boolean(state.running);
  $("make-hint").textContent = state.running ? "" : hint;
}

// ---------------------------------------------------------------------- jobs

function commonArgs() {
  const args = ["--per-hour", String(state.settings.perHour || 14)];
  if (state.settings.mode) args.push("--mode", state.settings.mode);
  if (state.settings.profile) args.push("--profile", state.settings.profile);
  return [...args, ...gamesArgs()];
}

function gamesArgs() {
  return state.settings.findGames === false ? ["--no-games"] : [];
}

async function make() {
  let args;
  if (state.source === "twitch") {
    args = ["run", "--vod", state.vod.url, "--out", state.settings.rawFolder,
            "--name", $("stem").value.trim(), ...commonArgs()];
  } else {
    args = ["run", "--file", state.video, ...commonArgs()];
    const url = $("match-url").value.trim() || $("match").value;
    if (url) args.push("--vod", url);
  }
  await startJob("run", args, "Making the list");
}

async function startJob(kind, args, title) {
  state.running = kind;
  state.steps = [];
  state.result = null;
  state.lastError = "";
  state.started = Date.now();
  $("progress").hidden = false;
  $("progress-title").textContent = title;
  $("steps").replaceChildren();
  $("log").textContent = "";
  $("step-text").textContent = "Starting…";
  $("step-text").classList.remove("error");
  $("bar-fill").style.width = "0";
  $("stop").hidden = false;
  if (kind === "run") $("done").hidden = true;
  updateMake();
  const answer = await api.start(args);
  if (!answer.ok) {
    state.running = null;
    $("step-text").textContent = answer.text;
    updateMake();
  }
}

function renderSteps(currentId) {
  const list = $("steps");
  list.replaceChildren();
  let passed = true;
  for (const step of state.steps) {
    const item = document.createElement("li");
    if (step.id === currentId) {
      item.className = "now";
      passed = false;
    } else if (passed && currentId) {
      item.className = "done";
    }
    item.textContent = step.label;
    list.append(item);
  }
}

function onEvent(event) {
  switch (event.type) {
    case "steps":
      state.steps = event.steps;
      renderSteps(null);
      break;
    case "step": {
      if (state.steps.length) renderSteps(event.id);
      const label = state.steps.find((s) => s.id === event.id)?.label ?? event.id;
      $("step-text").textContent = event.text ? `${label}: ${event.text}` : label;
      const index = state.steps.findIndex((s) => s.id === event.id);
      const within = typeof event.fraction === "number" ? event.fraction : 0;
      const overall = state.steps.length
        ? (Math.max(index, 0) + within) / state.steps.length : within;
      $("bar-fill").style.width = `${Math.round(overall * 100)}%`;
      break;
    }
    case "log":
      $("log").textContent += `${event.text}\n`;
      break;
    case "error":
      state.lastError = event.text;
      break;
    case "done":
    case "sync":
      if (state.running === "sync") {
        if (event.type === "sync") $("sync-note").textContent = event.note_a;
      } else {
        state.result = event;
      }
      break;
    case "exit":
      finish(event);
      break;
    default:
      break;
  }
}

function finish(exit) {
  const kind = state.running;
  state.running = null;
  $("stop").hidden = true;
  const minutes = Math.round((Date.now() - state.started) / 60000);
  if (kind === "run" && state.result) {
    $("progress").hidden = true;
    $("done").hidden = false;
    const r = state.result;
    $("done-text").textContent = `${r.lines} moments` +
      (r.games ? `, ${r.games} games found` : "") + ` · took ${minutes} min · ${r.text}`;
    $("sync-note").textContent = "";
  } else if (kind === "setup" && exit.code === 0) {
    $("progress").hidden = true;
    checkStatus();
  } else if (kind === "sync" && exit.code === 0) {
    $("progress").hidden = true;
  } else {
    $("bar-fill").style.width = "0";
    if (exit.cancelled) {
      $("step-text").textContent = "Stopped.";
    } else {
      const text = state.lastError || (exit.error ? `Can't start the backend: ${exit.error}` : "")
        || exit.tail.join("\n");
      $("step-text").textContent = text || `Stopped (exit ${exit.code}).`;
      $("step-text").classList.add("error");
    }
  }
  updateMake();
}

setInterval(() => {
  if (!state.running) return;
  const s = Math.round((Date.now() - state.started) / 1000);
  $("elapsed").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}, 1000);

async function lineUp() {
  if (!state.result) return;
  const other = await api.chooseVideo(state.settings.rawFolder);
  if (!other) return;
  await startJob("sync", ["sync", state.result.video, other,
                          "--per-hour", String(state.settings.perHour || 14), ...gamesArgs()],
                 "Lining up the two VODs");
}

// ---------------------------------------------------------------------- settings

async function openSettings() {
  const s = state.settings;
  $("set-raw").value = s.rawFolder;
  $("set-home").value = s.dataHome;
  $("set-per-hour").value = s.perHour;
  $("set-games").checked = s.findGames !== false;
  $("set-theme").value = s.theme ?? "";
  const modes = state.status?.modes ?? ["fast", "balanced", "best"];
  const names = { fast: "Fast (one small model)", balanced: "Balanced (one larger model)",
                  best: "Best (three models check each other)" };
  $("set-mode").replaceChildren(new Option(
    `Default for this PC (${state.status?.default_mode ?? "?"})`, ""),
    ...modes.map((m) => new Option(names[m] ?? m, m)));
  $("set-mode").value = s.mode ?? "";
  const profiles = state.status?.profiles ?? [];
  $("set-profile").replaceChildren(new Option(
    `Usual (${state.status?.default_profile ?? "none"})`, ""),
    ...profiles.map((p) => new Option(p, p)));
  $("set-profile").value = s.profile ?? "";
  $("settings").showModal();
}

async function closeSettings() {
  if ($("settings").returnValue !== "save") return;
  const homeChanged = $("set-home").value.trim() !== state.settings.dataHome;
  const modeChanged = $("set-mode").value !== (state.settings.mode ?? "");
  state.settings = await api.saveSettings({
    rawFolder: $("set-raw").value.trim(), dataHome: $("set-home").value.trim(),
    perHour: Number($("set-per-hour").value) || 14, mode: $("set-mode").value,
    profile: $("set-profile").value, theme: $("set-theme").value,
    findGames: $("set-games").checked,
  });
  applyTheme(state.settings.theme);
  $("raw-folder").textContent = state.settings.rawFolder;
  if (homeChanged || modeChanged) checkStatus();
  if (state.vod) checkAlready();
}

// ---------------------------------------------------------------------- start

async function init() {
  state.settings = await api.settings();
  applyTheme(state.settings.theme);
  $("account").value = state.settings.account ?? "";
  $("raw-folder").textContent = state.settings.rawFolder;
  api.onEvent(onEvent);

  $("tab-twitch").addEventListener("click", () => selectTab("twitch"));
  $("tab-file").addEventListener("click", () => selectTab("file"));
  $("find-form").addEventListener("submit", findVods);
  $("stem").addEventListener("input", () => { checkAlready(); updateMake(); });
  $("change-raw").addEventListener("click", async () => {
    const folder = await api.chooseFolder(state.settings.rawFolder);
    if (!folder) return;
    state.settings = await api.saveSettings({ rawFolder: folder });
    $("raw-folder").textContent = folder;
    checkAlready();
  });
  $("choose-video").addEventListener("click", chooseVideo);
  $("use-it").addEventListener("click", useSameDay);
  $("make").addEventListener("click", make);
  $("stop").addEventListener("click", () => api.cancel());
  $("open-log").addEventListener("click", () => api.openLog());
  $("open-html").addEventListener("click", () => api.open(state.result.html));
  $("open-txt").addEventListener("click", () => api.open(state.result.text));
  $("show-folder").addEventListener("click", () => api.show(state.result.html));
  $("sync").addEventListener("click", lineUp);
  $("run-setup").addEventListener("click", () => {
    const args = ["setup"];
    if (state.settings.mode) args.push("--mode", state.settings.mode);
    startJob("setup", args, "First-time setup");
  });
  $("open-settings").addEventListener("click", openSettings);
  $("settings").addEventListener("close", closeSettings);
  $("pick-raw").addEventListener("click", async () => {
    const folder = await api.chooseFolder($("set-raw").value);
    if (folder) $("set-raw").value = folder;
  });
  $("pick-home").addEventListener("click", async () => {
    const folder = await api.chooseFolder($("set-home").value);
    if (folder) $("set-home").value = folder;
  });

  updateMake();
  await checkStatus();
  if (state.settings.account) findVods();
}

init();
