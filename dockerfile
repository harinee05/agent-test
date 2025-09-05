FROM python:3.11

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y supervisor
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000
EXPOSE 9000

CMD ["/usr/bin/supervisord", "-c", "/app/supervisord.conf"]
# CMD ["bash", "start.sh"]