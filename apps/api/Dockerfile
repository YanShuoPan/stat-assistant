FROM python:3.12-slim

WORKDIR /app

COPY apps/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/api/ /app/apps/api/
COPY packages/ /app/packages/

WORKDIR /app/apps/api

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
