from pathlib import Path
import json

from starlette.applications import Starlette
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.staticfiles import StaticFiles

N_FEATURES = 682  # False to use the full model

# Load the model and maybe shrink it, using top n features
# model_file = Path("models/model__N=79556_F=746_L=31.json")
# model_file = Path("models/model__N=1800_F=746_L=6.json")
model_file = Path("models/model__N=157472_F=746_L=31.json")  # 10-line snippets
model = json.loads(model_file.read_text())
if N_FEATURES:
    model["features"] = model["features"][:N_FEATURES]
    model["coef"] = [row[:N_FEATURES] for row in model["coef"]]


app = Starlette(
    routes=[
        Route("/model.json", lambda r: JSONResponse(model)),
    ],
)
app.add_middleware(GZipMiddleware)
app.mount("/", StaticFiles(directory="web", html=True))
