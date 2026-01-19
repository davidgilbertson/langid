window.langidPredict = (snippet, model) => {
  const languages = model.classes;

  const features = model.features.map((featureName) => snippet.includes(featureName));

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
  const bestIndex = scores.indexOf(maxScore);

  // Softmax
  const exps = scores.map((score) => Math.exp(score - maxScore));
  const sum = exps.reduce((total, value) => total + value, 0);
  const probs = exps.map((value) => value / sum);

  return {language: languages[bestIndex], prob: probs[bestIndex]};
};
