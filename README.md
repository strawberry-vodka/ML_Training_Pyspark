import express from "express";
import fs from "fs";
import { handleChat } from "../utils/llmHandler.js";
import { generateChatTitle } from "../services/ollamaService.js";
import { estimateTokens } from "../utils/summaryBuilder.js";
import {
  ensureChatMemoryTable,
  getChatMemory,
  saveChatMemory,
} from "../db/index.js";
const router = express.Router();

router.post("/", async (req, res) => {
  const { id: chatId, message } = req.body;
  try {
    console.log(chatId, message);
    let { memorySummary, unsummarizedTurns, unsummarizedTokenCount } =
      await getChatMemory(chatId);
    let {
      stream,
      memorySummary: updatedMemorySummary,
      unsummarizedTokenCount: updatedUnsummarisedTknCount,
      unsummarizedTurns: updatedUnsummarisedTurns,
    } = await handleChat(
      chatId,
      message,
      memorySummary,
      unsummarizedTurns,
      unsummarizedTokenCount
    );
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("Access-Control-Allow-Origin", "*");
    let fullResponse = "";
    for await (const chunk of stream) {
      fs.writeFileSync("chunkContent.json", JSON.stringify(chunk));
      if (chunk?.content) {
        console.log(chunk.content);
        // res.write(JSON.stringify({ "status": "in-progress", "data": [{ "type": "text", "text": chunk.content }] }))
        res.write(chunk.content);
        fullResponse += chunk.content;
      } else if (typeof chunk === "string") {
        res.write(
          JSON.stringify({
            status: "in-progress",
            data: [{ type: "text", text: chunk }],
          })
        );
        fullResponse += chunk;
      }
    }
    // fs.writeFileSync('fullChat.txt', JSON.stringify(fullResponse, undefined, 4))
    res.end();
    updatedUnsummarisedTurns.push({ role: "assistant", content: fullResponse });
    updatedUnsummarisedTknCount += estimateTokens(fullResponse);
    await saveChatMemory(
      chatId,
      updatedMemorySummary,
      updatedUnsummarisedTurns,
      updatedUnsummarisedTknCount
    );
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to process chat" });
  }
});

router.post("/generate-chat-title", async (req, res) => {
  const { message } = req.body;
  try {
    const chatTitle = await generateChatTitle(message);
    res.status(200).json(chatTitle);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to process chat" });
  }
});

export default router;
