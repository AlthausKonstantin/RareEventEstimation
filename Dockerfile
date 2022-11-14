FROM python:3.9
RUN apt-get update
RUN apt-get install -y vim	
WORKDIR .
COPY . .
RUN pip install -e .
COPY ./shell-scripts/* /usr/local/bin
RUN chmod +x /usr/local/bin/*
CMD ["pytest", "./test"]