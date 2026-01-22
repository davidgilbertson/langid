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
        const prediction = predict(snippet, model);
        langEl.textContent = `${prediction.language} (${(prediction.prob * 100).toFixed(1)}%)`;
        langEl.classList.remove("lang-pill--empty");
        modelSectionEl.hidden = false;
        modelViz(snippet, model);
      };

      snippetEl.addEventListener("input", handleInput);
      handleInput();
    });
