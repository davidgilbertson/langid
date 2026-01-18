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

function predict(snippet) {
  const languages = model.classes;

  const features = model.features.map((token) => snippet.includes(token));

  const scores = [];

  for (let i = 0; i < languages.length; i++) {
    const row = (model.coef)[i];
    let score = (model.intercept)[i];
    for (let j = 0; j < features.length; j++) {
      score += features[j] * row[j];
    }
    scores.push(score);
  }

  const maxScore = Math.max(...scores);
  const bestIndex = scores.indexOf(maxScore);

  // Softmax
  const exps = scores.map((score) => Math.exp(score - maxScore));
  const sum = exps.reduce((total, value) => total + value, 0);
  const probs = exps.map((value) => value / sum);

  return {language: languages[bestIndex], prob: probs[bestIndex]};
}

window.langidPredictLanguage = async function langidPredictLanguage(snippet) {
  await ensureModelLoaded();
  return predict(snippet);
};

ensureModelLoaded();
