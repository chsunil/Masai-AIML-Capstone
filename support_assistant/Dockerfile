FROM python:3.12-slim

WORKDIR /app

# CPU-only torch: the default wheel pulls the CUDA stack, which is several GB
# and useless here since MiniLM runs fine on CPU.
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY . .

# Bake the MiniLM weights and the Chroma index into the image, so the container
# starts up without needing to reach Hugging Face at runtime.
RUN python ingest.py

ENV MOCK_LLM=1
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
