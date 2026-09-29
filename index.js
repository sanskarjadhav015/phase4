import express from "express";
import fs from "fs";
import dotenv from "dotenv";
import { PDFParse } from "pdf-parse";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";

dotenv.config();

const app = express();
const PORT = 5000;

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-large",
});

const upload = async () => {
  try {
    const pdfPath = "./knowledge.pdf";

    const buffer = fs.readFileSync(pdfPath);

    const parser = new PDFParse({
      data: buffer,
    });

    // 1. Extract PDF text
    const result = await parser.getText();

    // 2. Split text into chunks
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const docs = await splitter.createDocuments([result.text]);

    console.log("Number of chunks:", docs.length);

    // 3. Store chunks + embeddings in Qdrant
    const vectorStore = await QdrantVectorStore.fromDocuments(
      docs,
      embeddings,
      {
        url: process.env.QDRANT_URL,
        collectionName: "langchainjs-testing",
      }
    );

    console.log("Data stored in Qdrant ✅");

    await parser.destroy();

    return docs;

  } catch (error) {
    console.error("PDF Error:", error);
  }
};

app.get("/", (req, res) => {
  res.json({
    message: "Server is running 🚀",
  });
});

app.get("/upload", async (req, res) => {

  console.log("PDF processing started...");

  const docs = await upload();

  if (!docs) {
    return res.status(500).json({
      message: "PDF processing failed",
    });
  }

  res.json({
    message: "PDF processed successfully",
    chunks: docs.length,
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});