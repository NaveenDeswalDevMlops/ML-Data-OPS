FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt dashboard/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.enableCORS", "false"]
