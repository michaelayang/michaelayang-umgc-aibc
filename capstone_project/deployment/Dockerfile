# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app/static
WORKDIR /app/templates
WORKDIR /app


# Install system dependencies required for some Python packages if needed
# RUN apt-get update && apt-get install -y --no-install-recommends \
#    build-essential \
#    && rm -rf /var/lib/apt/lists/*


# Copy the requirements file into the container at /app
COPY ./requirements.txt .


# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


RUN ngrok http 8080 --url https://unsold-bolster-wildcat.ngrok-free.dev --pooling-enabled=true &


# Copy the rest of the backend application code into the container at /app
COPY main.py /app/.
COPY templates/index.html templates/.
COPY static/styles.css static/.
COPY MYangAICapstoneProjectUMGCPrototypeInput.json .
COPY MYangAICapstoneProjectUMGCForEvaluationString5.json .


# Make port 5000 available to the world outside this container
EXPOSE 5000


# Run main.py when the container launches
# Use uvicorn to run the FastAPI application
# --host 0.0.0.0 makes the server accessible from outside the container
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "5000"]
