import asyncio
import nest_asyncio
nest_asyncio.apply()

from openai import AsyncOpenAI
client = AsyncOpenAI(api_key="YOUR_KEY")

semaphore = asyncio.Semaphore(20)  # Limit to 20 concurrent requests

async def summarize_text(text):
    async with semaphore:
        response = await client.responses.create(
            model="gpt-4o-mini",
            instructions="Summarize the travel call transcript in 150–200 words...",
            input=text
        )
        return response.output_text

async def process_all(df, text_column):
    tasks = [summarize_text(t) for t in df[text_column]]
    results = await asyncio.gather(*tasks)
    df["masked_summary"] = results
    return df

# Usage in Jupyter
calls_df = await process_all(calls_df, "masked_text")
