{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "system",
      "content": "You are an expert conversational design analyst. Your job is to read masked call center transcripts and extract the conversation rules, decision-making steps, and tone used by human agents when helping customers in travel planning."
    },
    {import json
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# ====== Load Your Call Transcription Data ======
# This assumes your transcript JSON looks like:
# {
#   "scenario": "pre_booking_inquiry",
#   "transcript": "<PII-MASKED TRANSCRIPT TEXT>"
# }
with open("call_transcript.json", "r", encoding="utf-8") as f:
    call_data = json.load(f)

scenario = call_data.get("scenario", "unknown scenario")
transcript = call_data.get("transcript", "")

# ====== Create Prompt ======
system_prompt = (
    "You are an expert conversational design analyst. "
    "Your job is to read masked call center transcripts and extract the conversation rules, "
    "decision-making steps, and tone used by human agents when helping customers in travel planning."
)

user_prompt = f"""
Here is a {scenario} transcript. 
Extract the implicit rules followed by the agent and convert them into a reusable system prompt 
for a travel planning AI agent. 

Focus on:
1. Tone of conversation
2. Order of questions asked
3. Decision-making rules
4. Handling of positive, negative, or neutral responses

Return the output in the following format:

System Prompt:
ROLE: [role]
GOAL: [goal]
RULES: 
1. ...
2. ...
TONE: [tone]

[TRANSCRIPT START]
{transcript}
[TRANSCRIPT END]
"""

# ====== Send Request to OpenAI ======
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.3,
    max_tokens=1500
)

# ====== Extract & Save Output ======
generated_prompt = response.choices[0].message["content"]
print("\nGenerated System Prompt:\n")
print(generated_prompt)

# Optionally save to file
with open("generated_system_prompt.txt", "w", encoding="utf-8") as f:
    f.write(generated_prompt)

print("\nSystem prompt saved to 'generated_system_prompt.txt'")

      "role": "user",
      "content": "Here is a pre-booking travel inquiry transcript. Extract the implicit rules followed by the agent and convert them into a reusable system prompt for a travel destination suggestion AI. Focus on tone, question flow, and fallback handling.\n\n[TRANSCRIPT START]\n<Masked transcript here>\n[TRANSCRIPT END]"
    }
  ],
  "temperature": 0.3,
  "max_tokens": 1500
}
