const langEl = document.getElementById("lang");
const snippetEl = document.getElementById("snippet");

let model = null;

async function loadModel() {
  const res = await fetch("model.json", {cache: "no-store"});
  model = await res.json();

}

function softmax(scores) {
  const max = Math.max(...scores);
  const exps = scores.map((score) => Math.exp(score - max));
  const sum = exps.reduce((total, value) => total + value, 0);
  return exps.map((value) => value / sum);
}

function predict(snippet) {
  const tokens = model.features;
  const coef = model.coef;
  const bias = model.intercept;
  const languages = model.classes;

  const normalized = snippet.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const feats = tokens.map(token => normalized.includes(token));

  let bestIndex = 0;
  let bestScore = -Infinity;
  const scores = [];

  for (let i = 0; i < languages.length; i++) {
    const row = coef[i];
    let score = bias[i];
    for (let j = 0; j < feats.length; j++) {
      // TODO (@davidgilbertson): given feats[j] is boolean, no multiplication required?
      score += feats[j] * row[j];
    }
    scores.push(score);
    if (score > bestScore) {
      bestScore = score;
      bestIndex = i;
    }
  }

  const probs = softmax(scores);
  return {language: languages[bestIndex], prob: probs[bestIndex]};
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
  const prediction = predict(snippet);
  console.log("> prediction", prediction);
  langEl.textContent = `${prediction.language} (${(prediction.prob * 100).toFixed(1)}%)`;
});

await loadModel();
