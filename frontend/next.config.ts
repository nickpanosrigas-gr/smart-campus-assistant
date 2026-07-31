import type { NextConfig } from "next";
import dotenv from "dotenv";
import path from "path";

dotenv.config({ path: path.resolve(process.cwd(), '../.env') });

const nextConfig: NextConfig = {
  env: {
    API_URL: process.env.API_URL,
    OLLAMA_MODEL: process.env.OLLAMA_MODEL,
    OLLAMA_NUM_CTX: process.env.OLLAMA_NUM_CTX,
    GOOGLE_CLIENT_ID: process.env.GOOGLE_CLIENT_ID, 
  },
  
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://backend:8000/api/:path*' // Forwards to FastAPI
      },
      {
        source: '/ws/:path*',
        destination: 'http://backend:8000/ws/:path*'  // Forwards WebSockets to FastAPI
      }
    ];
  },
};

export default nextConfig;