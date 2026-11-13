#!/bin/bash -v

docker buildx build -t myang-chat-bot-jupyter .
docker run --interactive --tty -p 5000:5000 -v "$(pwd)/notebooks:/app/notebooks" myang-chat-bot-jupyter
