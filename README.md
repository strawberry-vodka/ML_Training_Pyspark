{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "system",
      "content": "You are an expert conversational design analyst. Your job is to read masked call center transcripts and extract the conversation rules, decision-making steps, and tone used by human agents when helping customers in travel planning."
    },
    {
      "role": "user",
      "content": "Here is a pre-booking travel inquiry transcript. Extract the implicit rules followed by the agent and convert them into a reusable system prompt for a travel destination suggestion AI. Focus on tone, question flow, and fallback handling.\n\n[TRANSCRIPT START]\n<Masked transcript here>\n[TRANSCRIPT END]"
    }
  ],
  "temperature": 0.3,
  "max_tokens": 1500
}
