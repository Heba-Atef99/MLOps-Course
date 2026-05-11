FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /code

COPY . /code

RUN chmod +x /code

RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN pip install --no-cache-dir -e .

EXPOSE 8005

ENV PYTHONPATH "${PYTHONPATH}:/code"

CMD ["python", "main.py"]
