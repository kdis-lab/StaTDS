# Use an official Python runtime as a base image
FROM python:3.11-slim-buster

# Install any needed packages specified in requirements.txt
RUN pip install -i https://pypi.python.org/simple/ --no-cache-dir --default-timeout=100 statds[full-app]


# Make port 8050 available to the world outside this container
EXPOSE 8050

RUN printf "from statds import app\napp.start_app(host='0.0.0.0', port=8050)" > start_app.py

# Run app.py when the container launches
CMD ["python", "start_app.py"]