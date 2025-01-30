FROM python:3.9-slim

WORKDIR /app
COPY ./requirements.txt /app/
COPY ./tutorial.py /app/
COPY ./your_image.jpg /app/

RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "tutorial.py"]
