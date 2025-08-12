import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="YOUR_OPENAI_API_KEY")

async def summarize_text(text):
    response = await client.responses.create(
        model="gpt-4o-mini",  # Faster and cheaper than gpt-4o
        instructions=(
            "You are a helpful AI assistant working for a travel booking company - CheapOAir. "
            "Summarize the following call transcript in 150-200 words, capturing all key issues. "
            "If the customer mentions a transaction number, booking number, or other key PII-masked details, include them."
        ),
        input=text
    )
    return response.output_text

async def process_all(transcripts):
    tasks = [summarize_text(t) for t in transcripts]
    return await asyncio.gather(*tasks)

if __name__ == "__main__":
    # Example: Load 10 transcripts from text files
    transcripts = []
    for i in range(1, 11):
        with open(f"transcript_{i}.txt", "r", encoding="utf-8") as f:
            transcripts.append(f.read())

    results = asyncio.run(process_all(transcripts))

    # Save results
    for idx, summary in enumerate(results, start=1):
        with open(f"summary_{idx}.txt", "w", encoding="utf-8") as f:
            f.write(summary)

    print("✅ All transcripts processed in parallel!")
