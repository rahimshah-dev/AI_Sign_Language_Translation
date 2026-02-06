import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20";

const revealItems = document.querySelectorAll("[data-reveal]");
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const statusEl = document.getElementById("status");
const predictionEl = document.getElementById("prediction");

const ctx = canvas.getContext("2d");

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.2 }
);

revealItems.forEach((item) => observer.observe(item));

let modelData = null;
let handLandmarker = null;
let lastVideoTime = -1;

const MODEL_URL = "/hand_landmarker.task";
const CENTROID_URL = "/model.json";

const labelFallback = (label) => label;

function updateStatus(message) {
  if (statusEl) statusEl.textContent = message;
}

function updatePrediction(label) {
  if (predictionEl) predictionEl.textContent = label ?? "--";
}

function resizeCanvas() {
  if (!video.videoWidth || !video.videoHeight) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

function normalizeFeatureLength(vec, expectedLen) {
  if (!expectedLen || expectedLen <= 0) return vec;
  if (vec.length < expectedLen) {
    return vec.concat(new Array(expectedLen - vec.length).fill(0));
  }
  if (vec.length > expectedLen) {
    return vec.slice(0, expectedLen);
  }
  return vec;
}

function predictLabel(features) {
  if (!modelData) return "--";
  const { centroids, labels, labelMap, featureLength } = modelData;
  const vec = normalizeFeatureLength(features, featureLength);
  let bestIndex = 0;
  let bestScore = Number.POSITIVE_INFINITY;
  for (let i = 0; i < centroids.length; i += 1) {
    const centroid = centroids[i];
    let sum = 0;
    for (let j = 0; j < centroid.length; j += 1) {
      const diff = (vec[j] ?? 0) - centroid[j];
      sum += diff * diff;
    }
    if (sum < bestScore) {
      bestScore = sum;
      bestIndex = i;
    }
  }
  const label = labels[bestIndex];
  return (labelMap && labelMap[label]) || labelFallback(label);
}

async function setupModel() {
  updateStatus("Loading classifier…");
  const response = await fetch(CENTROID_URL, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Unable to load model.json");
  }
  modelData = await response.json();
}

async function setupHandLandmarker() {
  updateStatus("Loading hand tracker…");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.3,
    minHandPresenceConfidence: 0.3,
    minTrackingConfidence: 0.3,
  });
}

async function setupCamera() {
  updateStatus("Requesting camera access…");
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);
}

function drawLandmarks(landmarks) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!landmarks.length) return;

  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;

  ctx.fillStyle = "rgba(42, 127, 125, 0.9)";

  landmarks.forEach((hand) => {
    hand.forEach((point) => {
      const x = point.x * canvas.width;
      const y = point.y * canvas.height;
      minX = Math.min(minX, point.x);
      minY = Math.min(minY, point.y);
      maxX = Math.max(maxX, point.x);
      maxY = Math.max(maxY, point.y);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    });
  });

  ctx.strokeStyle = "rgba(226, 111, 61, 0.9)";
  ctx.lineWidth = 3;
  ctx.strokeRect(
    minX * canvas.width - 8,
    minY * canvas.height - 8,
    (maxX - minX) * canvas.width + 16,
    (maxY - minY) * canvas.height + 16
  );
}

function extractFeatures(landmarks) {
  const features = [];
  landmarks.forEach((hand) => {
    const xs = hand.map((p) => p.x);
    const ys = hand.map((p) => p.y);
    const minX = Math.min(...xs);
    const minY = Math.min(...ys);
    hand.forEach((point) => {
      features.push(point.x - minX, point.y - minY);
    });
  });
  return features;
}

function detectLoop() {
  if (!handLandmarker || video.readyState < 2) {
    requestAnimationFrame(detectLoop);
    return;
  }

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const nowMs = performance.now();
    const results = handLandmarker.detectForVideo(video, nowMs);
    if (results.landmarks && results.landmarks.length > 0) {
      drawLandmarks(results.landmarks);
      const features = extractFeatures(results.landmarks);
      const label = predictLabel(features);
      updatePrediction(label);
      updateStatus("Hand detected");
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      updatePrediction("--");
      updateStatus("Show your hand to the camera");
    }
  }

  requestAnimationFrame(detectLoop);
}

async function init() {
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      updateStatus("Camera not supported in this browser.");
      return;
    }
    await setupModel();
    await setupHandLandmarker();
    await setupCamera();
    updateStatus("Ready");
    detectLoop();
  } catch (error) {
    console.error(error);
    updateStatus("Failed to start. Check console for details.");
  }
}

init();
