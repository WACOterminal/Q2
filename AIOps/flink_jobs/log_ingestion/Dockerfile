# Use an official Flink image as a parent image
FROM flink:1.15.4-scala_2.12-java11

# Set the working directory
WORKDIR /opt/flink

# Install Python and pip
RUN apt-get update -y && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy the job script
COPY job.py .

# Download Flink connectors
# In a real scenario, you might have a more robust way to handle these dependencies.
RUN mkdir -p /opt/flink/connectors && \
    wget -P /opt/flink/connectors https://repo.maven.apache.org/maven2/org/apache/flink/flink-sql-connector-pulsar/1.15.4/flink-sql-connector-pulsar-1.15.4.jar && \
    wget -P /opt/flink/connectors https://repo.maven.apache.org/maven2/org/apache/flink/flink-sql-connector-elasticsearch7/1.15.4/flink-sql-connector-elasticsearch7-1.15.4.jar

# Set the default command to run the job
CMD ["python3", "job.py"] 