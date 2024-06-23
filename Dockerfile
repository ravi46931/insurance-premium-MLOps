FROM python:3.11-slim-buster
WORKDIR /app
COPY . ./
ADD requirements_app.txt requirements_app.txt 
RUN pip install -r requirements_app.txt
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py"]