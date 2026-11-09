import uvicorn
from fastapi import FastAPI
from routers import search

app = FastAPI()
app.include_router(search.router)


@app.get("/")
async def health():
    return {"message": "healthy"}


def main():
    uvicorn.run(app, host="127.0.0.1", port=8085, reload=False, log_level="info")


if __name__ == "__main__":
    main()
