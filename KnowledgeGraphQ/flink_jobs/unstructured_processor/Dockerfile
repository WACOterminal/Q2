
# Use the official Flink image as a base
FROM apache/flink:1.15.2-scala_2.12-java11

# Set the working directory in the container
WORKDIR /opt/flink

# Copy the requirements file from the KnowledgeGraphQ service directory
COPY KnowledgeGraphQ/requirements.txt /opt/flink/requirements.txt

# Install Python and dependencies
RUN apt-get update -y && \
    apt-get install -y python3 python3-pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy the Flink job script into the container
COPY KnowledgeGraphQ/flink_jobs/unstructured_processor/job.py /opt/flink/

# Set the entrypoint to run the Flink Python job
# The job will be submitted to the Flink cluster by an external script or manually.
# This entrypoint is more for defining the job's location.
ENTRYPOINT ["flink", "run", "-py", "/opt/flink/job.py"] 