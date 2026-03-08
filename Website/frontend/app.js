function initApp() {
  const form = document.getElementById("detect-form");
  const statusPanel = document.getElementById("status");
  const resultFrame = document.getElementById("resultFrame") || document.getElementById("resultGif");
  const resultTable = document.getElementById("resultTable");
  const submitBtn = document.getElementById("submitBtn");
  const playBtn = document.getElementById("playBtn");
  const pauseBtn = document.getElementById("pauseBtn");
  const frameSlider = document.getElementById("frameSlider");
  const frameLabel = document.getElementById("frameLabel");
  const folderPathInput = document.getElementById("folderPath");
  const checkpointPathInput = document.getElementById("checkpointPath");
  const thresholdInput = document.getElementById("threshold");

  const player = {
    frames: [],
    currentIndex: 0,
    timerId: null,
    fps: 12,
  };

  function setStatus(text, isError = false) {
    if (!statusPanel) {
      return;
    }
    statusPanel.textContent = text;
    statusPanel.style.color = isError ? "#b00020" : "#1a2135";
  }

  function toFixedArray(arr, digits = 2) {
    return arr.map((x) => Number(x).toFixed(digits)).join(", ");
  }

  function renderTable(detections) {
    if (!resultTable) {
      return;
    }
    resultTable.innerHTML = "";
    if (!detections || detections.length === 0) {
      const row = document.createElement("tr");
      row.innerHTML = '<td colspan="5">No nodules above the threshold.</td>';
      resultTable.appendChild(row);
      return;
    }

    detections.forEach((item, idx) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${idx + 1}</td>
        <td>${item.probability.toFixed(4)}</td>
        <td>${toFixedArray(item.center_zyx)}</td>
        <td>${toFixedArray(item.center_xyz_mm)}</td>
        <td>${item.bbox_zyx.join(", ")}</td>
      `;
      resultTable.appendChild(row);
    });
  }

  function stopPlayback() {
    if (player.timerId !== null) {
      window.clearInterval(player.timerId);
      player.timerId = null;
    }
  }

  function updateControlState() {
    const hasFrames = player.frames.length > 0;
    if (playBtn) {
      playBtn.disabled = !hasFrames;
    }
    if (pauseBtn) {
      pauseBtn.disabled = !hasFrames;
    }
    if (frameSlider) {
      frameSlider.disabled = !hasFrames;
    }
  }

  function renderFrame(index) {
    if (!resultFrame) {
      return;
    }

    if (!player.frames.length) {
      resultFrame.removeAttribute("src");
      if (frameLabel) {
        frameLabel.textContent = "Frame 0 / 0";
      }
      return;
    }

    const safeIndex = ((index % player.frames.length) + player.frames.length) % player.frames.length;
    player.currentIndex = safeIndex;
    const frame = player.frames[safeIndex];
    if (!frame || !frame.url) {
      return;
    }
    resultFrame.src = frame.url;
    if (frameSlider) {
      frameSlider.value = String(safeIndex);
    }
    if (frameLabel) {
      frameLabel.textContent = `Frame ${safeIndex + 1} / ${player.frames.length} | Slice z=${frame.z_index}`;
    }
  }

  function startPlayback() {
    if (!player.frames.length || player.timerId !== null) {
      return;
    }

    player.timerId = window.setInterval(() => {
      renderFrame(player.currentIndex + 1);
    }, Math.round(1000 / player.fps));
  }

  function loadFrames(frames) {
    stopPlayback();
    player.frames = Array.isArray(frames) ? frames : [];
    player.currentIndex = 0;
    if (frameSlider) {
      frameSlider.min = "0";
      frameSlider.max = String(Math.max(player.frames.length - 1, 0));
      frameSlider.value = "0";
    }
    updateControlState();
    renderFrame(0);
    startPlayback();
  }

  if (playBtn) {
    playBtn.addEventListener("click", () => {
      startPlayback();
    });
  }

  if (pauseBtn) {
    pauseBtn.addEventListener("click", () => {
      stopPlayback();
    });
  }

  if (frameSlider) {
    frameSlider.addEventListener("input", (event) => {
      stopPlayback();
      renderFrame(Number(event.target.value));
    });
  }

  updateControlState();
  renderFrame(0);

  if (!form || !folderPathInput || !checkpointPathInput || !thresholdInput) {
    return;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const payload = {
      folder_path: folderPathInput.value.trim(),
      checkpoint_path: checkpointPathInput.value.trim(),
      probability_threshold: Number(thresholdInput.value),
    };

    if (submitBtn) {
      submitBtn.disabled = true;
    }
    stopPlayback();
    setStatus("Running inference...");

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Request failed");
      }

      const ts = Date.now();
      const frames = (data.frame_urls || []).map((frame) => ({
        url: `${frame.url}?t=${ts}`,
        z_index: frame.z_index,
      }));

      loadFrames(frames);
      renderTable(data.detections);
      setStatus(`Done: ${data.dicom_files} DICOM files, ${data.detections.length} detections.`);
    } catch (error) {
      loadFrames([]);
      setStatus(`Failed: ${error.message}`, true);
    } finally {
      if (submitBtn) {
        submitBtn.disabled = false;
      }
    }
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initApp);
} else {
  initApp();
}
