# frontend/Dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .

# Pass environment variables into the build process
ARG API_URL
ARG OLLAMA_MODEL
ARG OLLAMA_NUM_CTX
ENV API_URL=$API_URL
ENV OLLAMA_MODEL=$OLLAMA_MODEL
ENV OLLAMA_NUM_CTX=$OLLAMA_NUM_CTX

RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
COPY --from=builder /app/package.json ./
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/node_modules ./node_modules

EXPOSE 3000

CMD ["npm", "run", "start"]