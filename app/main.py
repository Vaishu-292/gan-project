from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schemas import GenerateRequest
from app.dependencies import get_gan_engine, GANEngine

app = FastAPI(title="STYLEGAN Classroom Project")


app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


@app.get("/")
def read_root(request: Request):
    """Renders the dashboard."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/generate")
def generate_art(request: GenerateRequest, engine: GANEngine = Depends(get_gan_engine)):
    image_url = engine.generate_image(request.seed,request.truncation)

    return {
        "status": "success",
        "seed": request.seed,
        "image_url": image_url,
        "mode": engine.mode
    }

                              