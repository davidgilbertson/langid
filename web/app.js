const langEl = document.getElementById("lang");
const snippetEl = document.getElementById("snippet");

let model = null;

async function loadModel() {
  const res = await fetch("model.json", {cache: "no-store"});
  model = await res.json();

}

function predict(snippet) {
  const tokens = model.features;
  const coef = model.coef;
  const bias = model.intercept;
  const languages = model.classes;

  const normalized = snippet.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  // const feats = tokens.map((token) => (normalized.includes(token) ? 1 : 0));
  const feats = tokens.map(token => normalized.includes(token));

  let bestIndex = 0;
  let bestScore = -Infinity;

  for (let i = 0; i < languages.length; i++) {
    const row = coef[i];
    let score = bias[i];
    for (let j = 0; j < feats.length; j++) {
      score += feats[j] * row[j];
    }
    if (score > bestScore) {
      bestScore = score;
      bestIndex = i;
    }
  }

  return languages[bestIndex];
}

snippetEl.addEventListener("input", () => {
  if (!model) {
    langEl.textContent = "model not loaded yet";
    return;
  }
  const snippet = snippetEl.value || "";
  if (snippet.length <= 5) {
    langEl.textContent = "—";
    return;
  }
  langEl.textContent = predict(snippet);
});

await loadModel();
