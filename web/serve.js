import express from "express";
import compression from "compression";
import {readFileSync} from "node:fs";

const N_FEATURES = 694; // set to null to use the full model
const model = JSON.parse(readFileSync("model.json", "utf8"));

if (N_FEATURES) {
  model.features = model.features.slice(0, N_FEATURES);
  model.coef = model.coef.map(row => row.slice(0, N_FEATURES));
}

express()
    .use(compression())
    .get("/model.json", (_, res) => res.json(model))
    .use(express.static("web"))
    .listen(8001);
