FROM apache/spark-py:v3.4.1

# Install additional Python dependencies
USER root
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    pulsar-client==3.2.0 \
    httpx==0.24.1 \
    mlxtend==0.22.0

# Copy job files
COPY job.py /opt/spark/work-dir/
COPY requirements.txt /opt/spark/work-dir/

WORKDIR /opt/spark/work-dir

USER spark

CMD ["/opt/spark/bin/spark-submit", \
     "--master", "spark://spark-master:7077", \
     "--deploy-mode", "client", \
     "--num-executors", "2", \
     "--executor-memory", "2g", \
     "--executor-cores", "2", \
     "--conf", "spark.sql.adaptive.enabled=true", \
     "--conf", "spark.sql.adaptive.coalescePartitions.enabled=true", \
     "job.py"] 