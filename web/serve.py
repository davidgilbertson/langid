from pathlib import Path
import json

from starlette.applications import Starlette
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.staticfiles import StaticFiles

N_FEATURES = 400  # False to use the full model

# Load the model and maybe shrink it, using top n features
model = json.loads(Path("model.json").read_text())
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
