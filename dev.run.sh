#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    # shellcheck disable=SC2046
    export $(grep -v '^#' .env | xargs)
fi

ollama pull $DOCUMENT_RAG_BOT_CHAT_MODEL__OLLAMA__0__DEPLOYMENTS__0__DEPLOYMENT_NAME
ollama pull $DOCUMENT_RAG_BOT_EMBEDDING_MODEL__OLLAMA__0__DEPLOYMENTS__0__DEPLOYMENT_NAME
ollama serve

# Start Streamlit application
poetry run streamlit run document_rag_bot/web/frontend.py --server.address $DOCUMENT_RAG_BOT_STREAMLIT_HOST --server.port $DOCUMENT_RAG_BOT_STREAMLIT_PORT --server.maxUploadSize 300 &

STREAMLIT_PID=$!

# Start ngrok tunnel
ngrok config add-authtoken $DOCUMENT_RAG_BOT_NGROK_AUTHTOKEN
nohup ngrok http $DOCUMENT_RAG_BOT_STREAMLIT_PORT > ngrok.log 2>&1 & until curl -s http://localhost:4040/api/tunnels | grep -o "https://[^\"]*"; do sleep 1; done

# Cleanup function to kill processes
cleanup() {
    echo "Cleaning up processes..."
    kill $STREAMLIT_PID
    pkill ngrok
}

trap cleanup EXIT

# Run the API
poetry run python -m document_rag_bot.web.api

# Wait for Streamlit process to finish
wait $STREAMLIT_PID
