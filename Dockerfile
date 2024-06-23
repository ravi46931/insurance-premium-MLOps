FROM python:3.11-slim-buster

COPY . ./
ADD requirements.txt requirements.txt 
RUN pip install -r requirements.txt
WORKDIR /app
ENTRYPOINT ["streamlit", "run", "app.py"]