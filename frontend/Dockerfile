# frontend/Dockerfile
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Install dependencies first (for caching)
COPY package.json package-lock.json ./
RUN npm install

# Copy the rest of the code
COPY . .

# Build the Next.js application for production
RUN npm run build

# Expose the Next.js port
EXPOSE 3000

# Start the Next.js server
CMD ["npm", "start"]