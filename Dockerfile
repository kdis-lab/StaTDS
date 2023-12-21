# Use an official Python runtime as a base image
FROM python:3.10-slim-buster

# Set the working directory in the container to /app
WORKDIR /Docker

# Copy the current directory contents into the container at /app
COPY . /Docker

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir StaTDS[full-app]

# Make port 8050 available to the world outside this container
EXPOSE 8050

RUN echo -e "from statds import app\napp.start_app(host='0.0.0.0', port=8050)" > start_app.py

# Run app.py when the container launches
CMD ["python", "start_app.py"]