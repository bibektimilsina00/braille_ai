from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api.braille import router as braille_router

app = FastAPI(title="Braille AI", description="Braille character recognition API")

# Include the braille prediction router
app.include_router(braille_router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "Braille AI API is running!"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


@app.get("/")
async def root():
    return {"message": "Welcome to Braille AI API!"}
