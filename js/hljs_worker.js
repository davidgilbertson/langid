/**
 * This file is intended to be run/accessed from Python, to test Highlight.js predictions
 * It returns predictions in 'The Stack' format.
 */
import readline from "node:readline";
import hljs from "highlight.js";

const hljsToStackMap = {
  c: "C",
  cpp: "C++",
  csharp: "C-Sharp",
  css: "CSS",
  dart: "Dart",
  diff: "Diff",
  go: "Go",
  graphql: "GraphQL",
  ini: "INI",
  java: "Java",
  javascript: "JavaScript",
  json: "JSON",
  kotlin: "Kotlin",
  less: "Less",
  lua: "Lua",
  makefile: "Makefile",
  xml: "XML",
  markdown: "Markdown",
  objectivec: "Objective-C",
  perl: "Perl",
  php: "PHP",
  plaintext: "Text",
  python: "Python",
  r: "R",
  ruby: "Ruby",
  rust: "Rust",
  scss: "SCSS",
  shell: "Shell",
  sql: "SQL",
  swift: "Swift",
  typescript: "TypeScript",
  vbnet: "Visual_Basic_.NET",
  wasm: "WebAssembly",
  yaml: "YAML",
};

const allowedHljsLanguages = Object.keys(hljsToStackMap);

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

rl.on("line", (line) => {
  if (!line.trim()) {
    return;
  }
  try {
    const {snippet} = JSON.parse(line);
    const result = hljs.highlightAuto(snippet ?? "", allowedHljsLanguages);
    const predicted = hljsToStackMap[result.language] ?? null;
    const payload = {
      predicted,
      hljsLanguage: result.language ?? null,
    };
    process.stdout.write(`${JSON.stringify(payload)}\n`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stdout.write(`${JSON.stringify({error: message})}\n`);
  }
});
