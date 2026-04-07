from fastapi import FastAPI

# import everything from your existing server file
from env import CodeReviewEnv
# (and other imports you already have)

app = FastAPI()


import uvicorn

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
# paste ALL your endpoints here:
# /health
# /tasks
# /reset
# /step
# etc
