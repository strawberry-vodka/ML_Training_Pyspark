import asyncio
import math
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="...")

# Estimate tokens per request
TOKENS_PER_REQUEST = 4000  
TPM_LIMIT = 200_000  
RPM_LIMIT = 3000  

MAX_REQUESTS_PER_MIN = min(RPM_LIMIT, TPM_LIMIT // TOKENS_PER_REQUEST)

semaphore = asyncio.Semaphore(MAX_REQUESTS_PER_MIN // 2)  # safer concurrency

async def summarize_text(text):
    async with semaphore:
        response = await client.responses.create(
            model="gpt-4o-mini",
            instructions="Summarize the transcript in 150-200 words...",
            input=text
        )
        return response.output_text

async def process_all(df, text_column):
    for i in range(0, len(df), MAX_REQUESTS_PER_MIN):
        batch = df.iloc[i:i+MAX_REQUESTS_PER_MIN]
        tasks = [summarize_text(t) for t in batch[text_column]]
        results = await asyncio.gather(*tasks)
        df.loc[batch.index, "masked_summary"] = results
        print(f"Processed batch {i // MAX_REQUESTS_PER_MIN + 1}")
        await asyncio.sleep(60)  # let quota reset
    return df["masked_summary"].tolist()
