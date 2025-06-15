FROM rapidsai/rapidsai:cuda11.8-runtime-ubuntu22.04-py3.10

WORKDIR /TopSpotifAI

COPY . .

RUN chmod +x entrypoint-prod.sh

# Install dependencies
RUN pip install -r requirements.txt

EXPOSE 8000

ENTRYPOINT ["sh", "entrypoint-prod.sh"]