const form = document.getElementById("detect-form");
const statusPanel = document.getElementById("status");
const resultGif = document.getElementById("resultGif");
const resultTable = document.getElementById("resultTable");
const submitBtn = document.getElementById("submitBtn");

function setStatus(text, isError = false) {
  statusPanel.textContent = text;
  statusPanel.style.color = isError ? "#b00020" : "#1a2135";
}

function toFixedArray(arr, digits = 2) {
  return arr.map((x) => Number(x).toFixed(digits)).join(", ");
}

function renderTable(detections) {
  resultTable.innerHTML = "";
  if (!detections || detections.length === 0) {
    const row = document.createElement("tr");
    row.innerHTML = `<td colspan="5">未检测到超过阈值的结节候选</td>`;
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

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    folder_path: document.getElementById("folderPath").value.trim(),
    checkpoint_path: document.getElementById("checkpointPath").value.trim(),
    probability_threshold: Number(document.getElementById("threshold").value),
  };

  submitBtn.disabled = true;
  setStatus("正在执行推理，请稍候...");

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "请求失败");
    }

    const ts = Date.now();
    resultGif.src = `${data.gif_url}?t=${ts}`;
    renderTable(data.detections);
    setStatus(`完成: ${data.dicom_files} 个 DICOM 文件, 检测到 ${data.detections.length} 个候选结节`);
  } catch (error) {
    setStatus(`失败: ${error.message}`, true);
  } finally {
    submitBtn.disabled = false;
  }
});
