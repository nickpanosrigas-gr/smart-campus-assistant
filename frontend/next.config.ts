import type { NextConfig } from "next";
import dotenv from "dotenv";
import path from "path";

// Attempt to load the .env file from the root directory (one level up)
// This handles local development perfectly. Inside Docker, variables are natively injected.
dotenv.config({ path: path.resolve(process.cwd(), '../.env') });

const nextConfig: NextConfig = {
  /* config options here */

  // Explicitly map the variables so Next.js embeds them into the browser bundle
  env: {
    API_URL: process.env.API_URL,
    OLLAMA_MODEL: process.env.OLLAMA_MODEL,
    OLLAMA_NUM_CTX: process.env.OLLAMA_NUM_CTX,
  }
};

export default nextConfig;