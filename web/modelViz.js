function formatNumber(value) {
  if (!Number.isFinite(value)) return "—";
  return value.toFixed(1);
}

function computeScores(snippet, model) {
  const languages = model.classes;
  const features = model.features.map((token) => snippet.includes(token));
  const scores = [];

  for (let i = 0; i < languages.length; i++) {
    const row = model.coef[i];
    let score = model.intercept[i];
    for (let j = 0; j < features.length; j++) {
      score += features[j] * row[j];
    }
    scores.push(score);
  }

  const maxScore = Math.max(...scores);
  const exps = scores.map((score) => Math.exp(score - maxScore));
  const sum = exps.reduce((total, value) => total + value, 0);
  const probs = exps.map((value) => value / sum);

  return {scores, probs, features};
}

window.langidModelViz = (snippet, model) => {
  const container = document.getElementById("model-viz");
  if (!container) return;

  container.innerHTML = "";

  const languages = model.classes;
  const {scores, probs, features} = computeScores(snippet || "", model);

  const wrapper = document.createElement("div");
  wrapper.className = "model-viz-wrap";

  const labelsTable = document.createElement("table");
  labelsTable.className = "model-viz-table model-viz-labels";
  const labelsHead = document.createElement("thead");
  const labelsHeadRow = document.createElement("tr");
  const labelsCorner = document.createElement("th");
  labelsCorner.className = "model-viz-corner";
  labelsCorner.textContent = "\u00A0";
  labelsHeadRow.appendChild(labelsCorner);
  labelsHead.appendChild(labelsHeadRow);
  labelsTable.appendChild(labelsHead);

  const labelsBody = document.createElement("tbody");
  const snippetLabelRow = document.createElement("tr");
  snippetLabelRow.className = "model-viz-snippet-row";
  const snippetLabel = document.createElement("th");
  snippetLabel.className = "model-viz-row-label";
  snippetLabel.textContent = "Snippet features:";
  snippetLabelRow.appendChild(snippetLabel);
  labelsBody.appendChild(snippetLabelRow);


  for (const language of languages) {
    const row = document.createElement("tr");
    const label = document.createElement("th");
    label.className = "model-viz-row-label";
    label.textContent = language;
    row.appendChild(label);
    labelsBody.appendChild(row);
  }
  labelsTable.appendChild(labelsBody);

  const weightsShell = document.createElement("div");
  weightsShell.className = "model-viz-weights-shell";
  const weightsTable = document.createElement("table");
  weightsTable.className = "model-viz-table model-viz-weights";

  const weightsHead = document.createElement("thead");
  const weightsHeadRow = document.createElement("tr");
  for (const feature of model.features) {
    const th = document.createElement("th");
    th.className = "model-viz-feature-header";
    const label = document.createElement("span");
    label.textContent = feature.replace(/\n/g, "\\n").replace(/\t/g, "\\t");
    th.appendChild(label);
    weightsHeadRow.appendChild(th);
  }
  weightsHead.appendChild(weightsHeadRow);
  weightsTable.appendChild(weightsHead);

  const weightsBody = document.createElement("tbody");
  const snippetWeightsRow = document.createElement("tr");
  snippetWeightsRow.className = "model-viz-snippet-row";
  for (let i = 0; i < features.length; i++) {
    const value = features[i];
    const cell = document.createElement("td");
    cell.className = "model-viz-number";
    cell.textContent = value ? "1" : "0";
    cell.classList.add(value ? "model-viz-feature-active" : "model-viz-feature-inactive");
    snippetWeightsRow.appendChild(cell);
  }
  weightsBody.appendChild(snippetWeightsRow);


  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);
  const scoreSpan = Math.max(1e-6, maxScore - minScore);

  for (let i = 0; i < languages.length; i++) {
    const row = document.createElement("tr");
    const weights = model.coef[i];
    for (let j = 0; j < weights.length; j++) {
      const weight = weights[j];
      const cell = document.createElement("td");
      cell.className = "model-viz-number";
      cell.textContent = formatNumber(weight);
      cell.classList.add(features[j] ? "model-viz-feature-active" : "model-viz-feature-inactive");
      row.appendChild(cell);
    }
    weightsBody.appendChild(row);
  }
  weightsTable.appendChild(weightsBody);
  weightsShell.appendChild(weightsTable);

  const metricsTable = document.createElement("table");
  metricsTable.className = "model-viz-table model-viz-metrics";
  const metricsHead = document.createElement("thead");
  const metricsHeadRow = document.createElement("tr");

  const biasHeader = document.createElement("th");
  biasHeader.className = "model-viz-metric-header model-viz-number";
  biasHeader.textContent = "Bias";
  metricsHeadRow.appendChild(biasHeader);

  const scoreHeader = document.createElement("th");
  scoreHeader.className = "model-viz-metric-header model-viz-number";
  scoreHeader.textContent = "Score";
  metricsHeadRow.appendChild(scoreHeader);

  const probHeader = document.createElement("th");
  probHeader.className = "model-viz-metric-header model-viz-number";
  probHeader.textContent = "Probability";
  metricsHeadRow.appendChild(probHeader);

  metricsHead.appendChild(metricsHeadRow);
  metricsTable.appendChild(metricsHead);

  const metricsBody = document.createElement("tbody");
  const snippetMetricsRow = document.createElement("tr");
  snippetMetricsRow.className = "model-viz-snippet-row";
  for (let i = 0; i < 3; i++) {
    const cell = document.createElement("td");
    cell.className = "model-viz-number model-viz-metric-cell";
    cell.textContent = "\u00A0";
    snippetMetricsRow.appendChild(cell);
  }
  metricsBody.appendChild(snippetMetricsRow);


  for (let i = 0; i < languages.length; i++) {
    const row = document.createElement("tr");

    const biasCell = document.createElement("td");
    biasCell.className = "model-viz-number model-viz-metric-cell";
    biasCell.textContent = formatNumber(model.intercept[i]);
    row.appendChild(biasCell);

    const scoreCell = document.createElement("td");
    scoreCell.className = "model-viz-number model-viz-metric-cell model-viz-score";
    scoreCell.textContent = formatNumber(scores[i]);
    const rawAlpha = (scores[i] - minScore) / scoreSpan;
    const scoreAlpha = Math.max(0, Math.min(1, rawAlpha));
    scoreCell.style.setProperty("background-color", `rgba(3 169 244 / ${scoreAlpha})`, "important");
    row.appendChild(scoreCell);

    const probCell = document.createElement("td");
    probCell.className = "model-viz-number model-viz-metric-cell";
    probCell.textContent = `${(probs[i] * 100).toFixed(1)}%`;
    row.appendChild(probCell);

    metricsBody.appendChild(row);
  }

  metricsTable.appendChild(metricsBody);

  wrapper.appendChild(labelsTable);
  wrapper.appendChild(weightsShell);
  wrapper.appendChild(metricsTable);
  container.appendChild(wrapper);
};
