FROM node:20-slim

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

ENV NODE_ENV=production
ENV PORT=3000
ENV FRONTEND_ORIGIN=http://localhost:3000

EXPOSE 3000

CMD ["npm", "run", "start"]
