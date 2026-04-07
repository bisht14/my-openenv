from fastapi import FastAPI

# import everything from your existing server file
from env import CodeReviewEnv
# (and other imports you already have)

app = FastAPI()

# paste ALL your endpoints here:
# /health
# /tasks
# /reset
# /step
# etc