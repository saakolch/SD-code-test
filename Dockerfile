# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV BASE_MODEL_PATH="sd_xl_base_1.0.safetensors"
ENV REFINER_MODEL_PATH="sd_xl_refiner_1.0.safetensors"

# Run main.py when the container launches
CMD ["python", "app/main.py", "app/config.json"]