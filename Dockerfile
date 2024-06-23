FROM python:3.11-slim-buster
LABEL org.opencontainers.image.title="My Dockerrrr Image"
LABEL org.opencontainers.image.description="This is a description of my Docker image."
LABEL org.opencontainers.image.category="Ravi Category"
EXPOSE 8080
COPY . ./
ADD requirements.txt requirements.txt 
RUN pip install -r requirements.txt
WORKDIR /app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]