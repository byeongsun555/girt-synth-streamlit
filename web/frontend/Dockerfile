FROM node:22-bookworm-slim

WORKDIR /workspace/multi_synth_streamlit/web/frontend

COPY web/frontend/package*.json ./
RUN npm ci

COPY web/frontend ./

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
