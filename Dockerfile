FROM continuumio/anaconda3
RUN mkdir -p /opt/audio-analysis
WORKDIR /opt/audio-analysis
EXPOSE 5000
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY run_server.py .
COPY sample.txt .
RUN mkdir RenderEngine
ADD RenderEngine/ RenderEngine/
ENV PORT 5000
ENTRYPOINT ["python"]
CMD ["run_server.py"]
