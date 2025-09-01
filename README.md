from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from routes.chat_routes import router as chat_router
import os

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # update if you want specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(chat_router, prefix="/api/chat")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)



from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
from utils.llmHandler import handle_chat
from services.ollamaService import generate_chat_title
from utils.summaryBuilder import estimate_tokens
from db import get_chat_memory, save_chat_memory

router = APIRouter()

@router.post("/")
async def chat_endpoint(request: Request):
    body = await request.json()
    chat_id = body.get("id")
    message = body.get("message")

    try:
        memory_summary, unsummarized_turns, unsummarized_token_count = await get_chat_memory(chat_id)

        result = await handle_chat(
            chat_id,
            message,
            memory_summary,
            unsummarized_turns,
            unsummarized_token_count
        )

        stream = result["stream"]
        updated_memory_summary = result["memorySummary"]
        updated_unsummarised_tkn_count = result["unsummarizedTokenCount"]
        updated_unsummarised_turns = result["unsummarizedTurns"]

        async def event_generator():
            full_response = ""
            async for chunk in stream:
                if isinstance(chunk, dict) and "content" in chunk:
                    yield chunk["content"]
                    full_response += chunk["content"]
                elif isinstance(chunk, str):
                    yield chunk
                    full_response += chunk

            updated_unsummarised_turns.append({"role": "assistant", "content": full_response})
            updated_unsummarised_tkn_count += estimate_tokens(full_response)

            await save_chat_memory(
                chat_id,
                updated_memory_summary,
                updated_unsummarised_turns,
                updated_unsummarised_tkn_count
            )

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        print(e)
        return JSONResponse(status_code=500, content={"error": "Failed to process chat"})


@router.post("/generate-chat-title")
async def generate_title(request: Request):
    body = await request.json()
    message = body.get("message")
    try:
        chat_title = await generate_chat_title(message)
        return JSONResponse(content=chat_title)
    except Exception as e:
        print(e)
        return JSONResponse(status_code=500, content={"error": "Failed to process chat"})
