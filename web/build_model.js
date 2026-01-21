// This loads a particular model, optionally reduces its size,
//  and saves it for the web demo to use
import {readFileSync, writeFileSync} from "node:fs";

const MODEL_FILE = "models/model__N=157472_F=746_L=31.json";
const N_FEATURES = 530; // set to null to use the full model

const model = JSON.parse(readFileSync(MODEL_FILE, "utf8"));

if (N_FEATURES) {
  model.features = model.features.slice(0, N_FEATURES);
  model.coef = model.coef.map(row => row.slice(0, N_FEATURES));
}

writeFileSync("web/public/model.json", JSON.stringify(model));
