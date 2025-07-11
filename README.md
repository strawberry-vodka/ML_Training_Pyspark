AI-Powered Chatbot for Travel Booking Platform: Project Approaches for Intent Classification, NER, and Retrieval-Augmented Generation (RAG)

📖 Introduction

You are building an AI-powered chatbot for a travel booking portal (similar to Kayak.com) that supports users with flight bookings and related queries. The chatbot is expected to handle two major interaction flows:

Pre-booking: Help users make new bookings and explore options.

Post-booking: Help users with modifications, cancellations, and queries related to seats, baggage, ancillaries, and general support.

Your project is divided into two Proof-of-Concepts (POCs):

✅ POC 1: Intent Classification & Named Entity Recognition (NER)

Classify user queries into specific travel-related intents.

Extract relevant structured entities (like booking ID, source, destination, date, seat, baggage info) from user messages.

✅ POC 2: Retrieval-Augmented Generation (RAG)

Handle FAQs and policy-based questions using document retrieval and LLM-based generation.

Currently, your only available data is call center transcription data, and you are exploring additional external sources and cost-effective modeling options.

🔹 Summary of Approaches

✨ POC 1: Intent Classification & NER

Approach 1: Rule-Based with Keyword Matching

Objective & Mechanism: Use hardcoded rules, regular expressions, and keyword lists to match phrases with intents and extract entities.

Data Required: Domain vocabulary, airport codes, airline names.

Modeling Technique: No ML involved; regex and pattern matching.

Deployment Complexity: Low (basic scripts or REST API).

Cost Involved: Very Low (initial setup and rule maintenance).

Example:

If message contains "cancel" and "flight" -> intent = "Cancel Booking"

Regex: "[A-Z0-9]{6}" to extract Booking ID

Approach 2: Prompt Engineering with Few-shot Learning

Objective & Mechanism: Use local LLMs with handcrafted few-shot prompts to classify intent and extract entities in JSON format.

Data Required: A small set of manually labeled examples from transcription data.

Modeling Technique: Few-shot prompting (no training); zero-shot fallback.

Deployment Complexity: Low-Medium (LLM API with FastAPI backend).

Cost Involved: Low (local LLMs like Mistral, Phi2; no API costs).

Example:

Classify the user intent:
User: "I need to cancel my flight to Delhi"
Intent: Cancel Booking

Extract entities:
Message: "Flight from Mumbai to Dubai on 20th August, Booking ID A1B2C3"
Output: {
  "source": "Mumbai",
  "destination": "Dubai",
  "travel_date": "20th August",
  "booking_id": "A1B2C3"
}

Approach 3: Supervised Learning using Annotated Transcriptions

Objective & Mechanism: Train separate models for intent classification and NER using annotated call data.

Data Required: 1K-10K annotated transcripts with labeled intents and entities.

Modeling Technique:

Intent: BERT + Classification Head

NER: spaCy, BERT-CRF, Flair

Deployment Complexity: Medium (model training + inference hosting).

Cost Involved: Moderate (labeling effort + training compute).

Example:

Labeled Text: "I want to add baggage to my booking."

Intent: "Baggage Related Query", Entity: {"baggage_info": "Yes"}

Approach 4: Hybrid (Rule-based + LLM or ML)

Objective & Mechanism: Use rules for high-confidence matches and fallback to ML/LLMs when ambiguous.

Data Required: Domain vocabulary + small labeled set.

Modeling Technique: Heuristics combined with prompt-based or model-based classification.

Deployment Complexity: Medium.

Cost Involved: Low-Medium.

Example:

Rule match: "Cancel Booking"

Else: LLM inference

🔹 POC 2: Retrieval-Augmented Generation (RAG)

Approach 1: Classic FAQ Bot using TF-IDF or BM25

Objective & Mechanism: Retrieve matching FAQ from indexed corpus using lexical similarity.

Data Required: Cleaned FAQ documents, booking policy documents.

Modeling Technique: BM25, TF-IDF, cosine similarity.

Deployment Complexity: Low-Medium (simple REST API with ElasticSearch/Faiss).

Cost Involved: Low.

Example:

User: "What is the baggage limit for domestic flights?"

Matched Answer: FAQ #3 from corpus

Approach 2: RAG Pipeline with LLM + Vector Store

Objective & Mechanism: Retrieve top-k context chunks from document DB (e.g., Pinecone/FAISS) and pass to LLM for response generation.

Data Required: Booking policy documents, help center content, chat logs.

Modeling Technique:

Embedding models (e.g., BGE, SBERT)

RAG architecture (LangChain, LlamaIndex)

Deployment Complexity: Medium-High (LLM + vector DB orchestration).

Cost Involved: Medium-High (embedding generation, hosting infra).

Example:

User: "Can I carry 10kg cabin baggage?"

LLM generates response using retrieved policy chunk.

Approach 3: Fine-tuned QA Model

Objective & Mechanism: Fine-tune a QA model (e.g., LLaMA2, BERT QA) using annotated questions + answers from internal sources.

Data Required: Annotated question-answer pairs, documents.

Modeling Technique: Fine-tuned transformer QA.

Deployment Complexity: High (training pipeline + model serving).

Cost Involved: High (GPU training + continuous maintenance).

Example:

User: "How do I cancel a flight?"

Model directly returns: "Log into your account, go to 'My Bookings'..."

🔹 External Data Sources You May Need

Airport and city code mappings

Airline names and codes

Flight schedule format

Booking policy documents (PDFs, HTML)

Help center content (FAQ pages)

User interaction logs

🚀 Final Recommendations

Start with prompt engineering for POC1 for speed and cost-efficiency.

Move to supervised models once labeled data is ready.

For POC2, begin with FAQ + RAG and scale to fine-tuning based on traffic and accuracy needs.

Let me know if you'd like to build out a timeline or implementation plan next.

