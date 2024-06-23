FROM python:3.11-slim-buster

EXPOSE 8080
COPY . ./
ADD requirements.txt requirements.txt 
RUN pip install -r requirements.txt
WORKDIR /app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]