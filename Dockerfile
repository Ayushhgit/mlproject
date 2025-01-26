# Base image with Python (choose version based on your project)
FROM python:3.8.0

# Set the working directory in the container
WORKDIR /app

COPY . /app

# Install Python dependencies
RUN apt update -y && apt install awscli -y

RUN pip install -r requirements.txt

# Define the command to run your project
# Replace `app.py` with the entry point of your project
CMD ["python3", "app.py"]
