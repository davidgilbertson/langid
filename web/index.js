import predict from "./predict.js";
import modelViz from "./modelViz.js";

const langEl = document.getElementById("lang");
const snippetEl = document.getElementById("snippet");
const modelSectionEl = document.getElementById("model-internals");

fetch("model.json", {cache: "no-store"})
    .then((res) => res.json())
    .then((model) => {
      const handleInput = () => {
        const snippet = snippetEl.value || "";
        if (snippet.length <= 5) {
          langEl.textContent = "—";
          langEl.classList.add("lang-pill--empty");
          modelSectionEl.hidden = true;
          modelViz("", model);
          return;
        }
        const start = performance.now();
        const prediction = predict(snippet, model);
        console.log(performance.now() - start);
        langEl.textContent = `${prediction.language} (${(prediction.prob * 100).toFixed(1)}%)`;
        langEl.classList.remove("lang-pill--empty");
        modelSectionEl.hidden = false;
        modelViz(snippet, model);
      };

      snippetEl.addEventListener("input", handleInput);
      handleInput();
    });
