import fs from "node:fs";
import path from "node:path";
import {performance} from "node:perf_hooks";
import {tableFromIPC} from "apache-arrow";
import {predict_language} from "./predict_language.js";

const datasetDir = "E:\\Datasets\\the_stack_select";
const dataFile = path.join(datasetDir, "data-00000-of-00001.arrow");
const buffer = fs.readFileSync(dataFile);
const table = tableFromIPC(buffer);

const languageColumn = table.getChild("Language");
const snippetColumn = table.getChild("Snippet");

const sampleCount = null;
const total = table.numRows;
const indices = sampleCount === null
    ? Array.from({length: total}, (_, idx) => idx)
    : (() => {
      const arr = [];
      const step = Math.max(1, Math.floor(total / sampleCount));
      for (let i = 0; i < total && arr.length < sampleCount; i += step) {
        arr.push(i);
      }
      return arr;
    })();

let correct = 0;
let totalScored = 0;
let totalTimeMs = 0;
const mistakeCounts = new Map();

for (const index of indices) {
  const snippet = snippetColumn.get(index);
  const actual = languageColumn.get(index);

  const start = performance.now();
  const predicted = predict_language(snippet);
  totalTimeMs += performance.now() - start;

  totalScored += 1;
  if (predicted === actual) {
    correct += 1;
  } else {
    const key = `${actual} -> ${predicted}`;
    mistakeCounts.set(key, (mistakeCounts.get(key) ?? 0) + 1);
  }
}

const accuracy = correct / totalScored;
const averageTimeMs = totalTimeMs / totalScored;

console.log(`Samples: ${totalScored}`);
console.log(`Accuracy: ${(accuracy * 100).toFixed(2)}%`);
console.log(`Avg time per prediction: ${averageTimeMs.toFixed(2)} ms`);

// console.log("Mistakes (actual -> predicted: count):");
// const sortedMistakes = Array.from(mistakeCounts.entries()).sort((a, b) => b[1] - a[1]);
// for (const [key, count] of sortedMistakes) {
//   console.log(`${key}: ${count}`);
// }
