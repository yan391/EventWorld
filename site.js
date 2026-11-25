"use strict";

const byId = id => document.getElementById(id);
const sceneButtons = [...document.querySelectorAll("[data-scene]")];
const video = byId("scene-source");
const canvas = byId("scene");
const view = canvas.getContext("2d", { alpha: false });
const viewport = byId("viewport");
const split = byId("split");
const seek = byId("seek");
const play = byId("play");
const mode = byId("depth-mode");
const note = byId("viewer-note");
const rgbPoster = byId("rgb-poster");
const depthPoster = byId("depth-poster");
const noteDefault = note.textContent;
const labels = { forecast: "EventWorld depth", baseline: "VGGT-World depth", truth: "Ground-truth depth" };
const quadrants = { forecast: [1, 1], baseline: [0, 1], truth: [1, 0] };
const frameRate = 25;
const frameCount = 240;
let scene = "urban";
let sourceScene = null;
let pendingFrame = null;
let animation = null;
let autoplayPending = true;
const videoFrameCallbacks = typeof video.requestVideoFrameCallback === "function";

function press(buttons, name, value) {
  for (const button of buttons) button.setAttribute("aria-pressed", button.dataset[name] === value);
}

function setPlayLabel(playing) {
  play.setAttribute("aria-label", playing ? "Pause driving sequence" : "Play driving sequence");
  play.classList.toggle("is-active", playing);
}

function updatePosition(frame) {
  seek.value = frame;
  const elapsed = (frame / frameRate).toFixed(1);
  const total = (frameCount / frameRate).toFixed(1);
  seek.setAttribute("aria-valuetext", `${elapsed} of ${total} seconds`);
  byId("time").value = `${elapsed} / ${total} s`;
}

function loadSequence() {
  if (sourceScene === scene) return;
  sourceScene = scene;
  video.preload = "auto";
  video.src = `assets/${scene}.mp4`;
  video.load();
}

function jumpToFrame(frame) {
  pendingFrame = frame;
  loadSequence();
  if (video.readyState < HTMLMediaElement.HAVE_METADATA) return;
  // Sample inside the target frame to avoid rounding onto the preceding one.
  video.currentTime = (frame + .15) / frameRate;
  pendingFrame = null;
}

function renderFrame() {
  if (sourceScene !== scene || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.seeking) return;
  const w = video.videoWidth / 2;
  const h = video.videoHeight / 2;
  const pixelWidth = viewport.clientWidth * (window.devicePixelRatio || 1);
  const width = Math.min(w * 2, Math.max(2, Math.round(pixelWidth / 2) * 2));
  const height = width / 2;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  view.imageSmoothingQuality = "high";
  const edge = Math.round(width * Number(split.value) / 100);
  const [column, row] = quadrants[mode.value];
  view.drawImage(video, 0, 0, w, h, 0, 0, width, height);
  view.save();
  view.beginPath();
  view.rect(edge, 0, width - edge, height);
  view.clip();
  view.drawImage(video, column * w, row * h, w, h, 0, 0, width, height);
  view.restore();
  viewport.classList.add("is-playing");
  updatePosition(Math.min(frameCount - 1, Math.floor(video.currentTime * frameRate + .0001)));
}

function requestPlaybackFrame() {
  if (animation !== null) return;
  animation = videoFrameCallbacks ? video.requestVideoFrameCallback(playbackLoop) : requestAnimationFrame(playbackLoop);
}

function stopPlaybackFrame() {
  if (animation === null) return;
  if (videoFrameCallbacks) video.cancelVideoFrameCallback(animation);
  else cancelAnimationFrame(animation);
  animation = null;
}

function playbackLoop() {
  animation = null;
  renderFrame();
  if (!video.paused && !video.ended) requestPlaybackFrame();
}

async function beginPlayback() {
  loadSequence();
  if (video.ended) video.currentTime = 0;
  try {
    await video.play();
    note.textContent = noteDefault;
  } catch (error) {
    if (error.name !== "AbortError") note.textContent = "Playback could not start. Try Play again.";
  }
}

play.addEventListener("click", () => {
  autoplayPending = false;
  if (video.paused) beginPlayback();
  else video.pause();
});
video.addEventListener("play", () => {
  setPlayLabel(true);
  requestPlaybackFrame();
});
function handlePlaybackStop() {
  stopPlaybackFrame();
  setPlayLabel(false);
}
video.addEventListener("pause", handlePlaybackStop);
video.addEventListener("ended", handlePlaybackStop);
video.addEventListener("loadedmetadata", () => {
  if (pendingFrame !== null) jumpToFrame(pendingFrame);
});
for (const name of ["loadeddata", "seeked", "ended"]) video.addEventListener(name, renderFrame);
new ResizeObserver(() => {
  if (viewport.classList.contains("is-playing")) renderFrame();
}).observe(viewport);
video.addEventListener("error", () => {
  setPlayLabel(false);
  note.textContent = "The driving clip could not load. Try switching scenes.";
});

for (const button of sceneButtons) {
  button.addEventListener("click", () => {
    if (button.dataset.scene === scene) return;
    const wasPlaying = !video.paused;
    video.pause();
    scene = button.dataset.scene;
    pendingFrame = null;
    viewport.classList.remove("is-playing");
    rgbPoster.src = `assets/${scene}-rgb.webp`;
    depthPoster.src = `assets/${scene}-${mode.value}.webp`;
    press(sceneButtons, "scene", scene);
    note.textContent = noteDefault;
    updatePosition(0);
    if (wasPlaying) beginPlayback();
  });
}

split.addEventListener("input", () => {
  viewport.style.setProperty("--split", `${split.value}%`);
  split.setAttribute("aria-valuetext", `${split.value}% RGB`);
  if (viewport.classList.contains("is-playing")) renderFrame();
});
seek.addEventListener("input", () => {
  autoplayPending = false;
  const frame = Number(seek.value);
  video.pause();
  updatePosition(frame);
  jumpToFrame(frame);
});
mode.addEventListener("change", () => {
  depthPoster.src = `assets/${scene}-${mode.value}.webp`;
  byId("depth-label").textContent = labels[mode.value];
  canvas.setAttribute("aria-label", `Synchronized comparison of target-time RGB and ${labels[mode.value]}`);
  if (viewport.classList.contains("is-playing")) renderFrame();
});

const viewObserver = new IntersectionObserver(([entry]) => {
  if (!entry.isIntersecting) {
    video.pause();
  } else if (autoplayPending) {
    autoplayPending = false;
    beginPlayback();
  }
});
viewObserver.observe(viewport);

const horizonButtons = [...document.querySelectorAll("[data-horizon]")];
const galleryScene = byId("gallery-scene");
const galleryStatus = byId("gallery-status");
let selectedHorizon = "200";
let galleryRevision = 0;

async function showHorizon() {
  const revision = ++galleryRevision;
  const prefix = `assets/horizons/scene${galleryScene.value}-${selectedHorizon}-`;
  const kinds = ["rgb", "forecast", "reference"];
  try {
    const images = await Promise.all(kinds.map(async kind => {
      const image = new Image();
      image.src = prefix + kind + ".webp";
      await image.decode();
      return image;
    }));
    if (revision !== galleryRevision) return;
    kinds.forEach((kind, index) => { byId("gallery-" + kind).src = images[index].src; });
    galleryStatus.textContent = `${galleryScene.selectedOptions[0].text}, +${selectedHorizon} ms.`;
    galleryStatus.className = "sr-only";
  } catch {
    if (revision !== galleryRevision) return;
    galleryStatus.className = "gallery-error";
    galleryStatus.textContent = "Could not load this comparison. The previous frames are still shown. Select a horizon to retry.";
  }
}
for (const button of horizonButtons) {
  button.addEventListener("click", () => {
    selectedHorizon = button.dataset.horizon;
    press(horizonButtons, "horizon", selectedHorizon);
    showHorizon();
  });
}
galleryScene.addEventListener("change", showHorizon);

const results = {
  depth: {
    asset: "eventscape-depth.webp", width: 1992, height: 780,
    alt: "Two EventScape examples with target-time inputs, depth and signed relative errors at 200 and 800 milliseconds.",
    text: "EventScape, 200 and 800 ms. Blue marks underestimation and red marks overestimation after scale alignment.",
  },
  cloud: {
    asset: "eventscape-cloud.webp", width: 1992, height: 656,
    alt: "EventScape point-cloud forecasts and competing methods at 200, 400 and 800 milliseconds.",
    text: "EventScape, 200, 400 and 800 ms. References are back-projected from depth annotations. At 800 ms, EventWorld reaches 42.75% F-score at 1% of the reference cloud radius, compared with 34.41% for DriveLaW (RGB).",
  },
  pose: {
    asset: "m3ed-poses.webp", width: 1434, height: 834,
    alt: "Measured and predicted camera poses from EventWorld and VGGT-World RGB on two M3ED sequences.",
    text: "M3ED, up to 800 ms. Without fine-tuning, EventWorld achieves 0.0973 m absolute trajectory error on the quantitative recording, compared with 0.1039 m for DriveLaW (RGB), after per-window similarity alignment.",
  },
};
const resultButtons = [...document.querySelectorAll("[data-result]")];
for (const button of resultButtons) {
  button.addEventListener("click", () => {
    const name = button.dataset.result;
    const item = results[name];
    const url = `assets/${item.asset}`;
    const image = byId("result-image");
    image.alt = item.alt;
    image.width = item.width;
    image.height = item.height;
    image.src = url;
    byId("result-link").href = url;
    byId("result-figure").dataset.view = name;
    byId("result-caption").textContent = item.text;
    press(resultButtons, "result", name);
  });
}

const plot = byId("plot");
const plotNote = byId("plot-note");
const datasetButtons = [...document.querySelectorAll("[data-dataset]")];
const datasetNotes = {
  DSEC: "DSEC low-light recording, 83 windows. The depth reference comes from calibrated disparity, with valid points within 30 m and visible to both cameras.",
  PEOD: "PEOD high-speed recording, 73 windows. EventVGGT produces the depth pseudo-references from target-time events. The evaluation uses finite positive pseudo-depth.",
};
const chartSeries = [
  ["EventWorld", "#0b7772"],
  ["VGGT-World (RGB)", "#5c63a1"],
  ["VGGT-World (Event)", "#bd5a28"],
];
let dataset = "DSEC";

function plotCurves() {
  const { width, height } = plot.getBoundingClientRect();
  if (width === 0 || height === 0) return;
  const ratio = window.devicePixelRatio || 1;
  plot.width = Math.round(width * ratio);
  plot.height = Math.round(height * ratio);
  const ctx = plot.getContext("2d");
  ctx.scale(ratio, ratio);
  const left = 45;
  const top = 27;
  const right = width - 20;
  const bottom = height - 48;
  const xFor = i => left + (right - left) * i / 7;
  const yFor = value => bottom - (bottom - top) * value / .42;
  ctx.font = '600 14px "Segoe UI", Arial, sans-serif';
  ctx.fillStyle = "#40595a";
  ctx.textBaseline = "middle";
  ctx.fillText("AbsRel ↓", 0, 9);
  for (const tick of [0, .1, .2, .3, .4]) {
    const y = yFor(tick);
    ctx.strokeStyle = "#dce4f1";
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(tick.toFixed(1), left - 7, y);
  }
  ctx.textAlign = "center";
  for (let i = 0; i < 8; i++) {
    if (width < 365 && i % 2 === 0 && i !== 0) continue;
    ctx.fillText(100 * (i + 1), xFor(i), bottom + 15);
  }
  ctx.fillText("Forecast horizon (ms)", (left + right) / 2, height - 10);

  for (const [name, color] of [...chartSeries].reverse()) {
    const values = transferCurves[dataset][name];
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = name === "EventWorld" ? 3 : 2;
    ctx.beginPath();
    values.forEach((value, i) => {
      if (i === 0) ctx.moveTo(xFor(i), yFor(value));
      else ctx.lineTo(xFor(i), yFor(value));
    });
    ctx.stroke();
    for (const [i, value] of values.entries()) {
      ctx.beginPath();
      ctx.arc(xFor(i), yFor(value), 3.25, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

function updateTable() {
  const rows = Array.from({ length: 8 }, (_, index) => {
    const row = document.createElement("tr");
    const heading = document.createElement("th");
    heading.scope = "row";
    heading.textContent = `${(index + 1) * 100} ms`;
    row.append(heading);
    for (const [name] of chartSeries) {
      const cell = document.createElement("td");
      cell.textContent = transferCurves[dataset][name][index].toFixed(4);
      row.append(cell);
    }
    return row;
  });
  byId("data-table").tBodies[0].replaceChildren(...rows);
}

for (const button of datasetButtons) {
  button.addEventListener("click", () => {
    dataset = button.dataset.dataset;
    press(datasetButtons, "dataset", dataset);
    plotNote.textContent = datasetNotes[dataset];
    plot.setAttribute("aria-label", `AbsRel by forecast horizon on ${dataset}. The data table below provides exact plotted values.`);
    plotCurves();
    updateTable();
  });
}
new ResizeObserver(plotCurves).observe(plot);
document.fonts.ready.then(plotCurves);
updateTable();
