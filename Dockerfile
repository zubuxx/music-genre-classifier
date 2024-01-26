FROM  --platform=linux/x86_64 python:3.11.7
WORKDIR /app
RUN pip install --upgrade pip
COPY ./requirements.txt /app
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
ENV FLASK_APP=app.py
ENV FLASK_DEBUG=true
CMD ["flask", "run", "--host", "0.0.0.0"]

