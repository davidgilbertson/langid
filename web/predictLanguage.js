let model = null;
let modelPromise = null;

async function loadModel() {
  const res = await fetch("model.json", {cache: "no-store"});
  model = await res.json();
}

function ensureModelLoaded() {
  if (!modelPromise) modelPromise = loadModel();
  return modelPromise;
}

function softmax(scores) {
  const max = Math.max(...scores);
  const exps = scores.map((score) => Math.exp(score - max));
  const sum = exps.reduce((total, value) => total + value, 0);
  return exps.map((value) => value / sum);
}

function predict(snippet) {
  const languages = model.classes;

  const normalized = snippet.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const feats = model.features.map((token) => normalized.includes(token));

  let bestIndex = 0;
  let bestScore = -Infinity;
  const scores = [];

  for (let i = 0; i < languages.length; i++) {
    const row = (model.coef)[i];
    let score = (model.intercept)[i];
    for (let j = 0; j < feats.length; j++) {
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

window.langidPredictLanguage = async function langidPredictLanguage(snippet) {
  await ensureModelLoaded();
  return predict(snippet);
};

ensureModelLoaded();
