# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install any dependencies in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire current directory into the container at /app
COPY . /app/

# Expose port 8501 for Streamlit to run
EXPOSE 8501

# Command to run your app using Streamlit
CMD ["streamlit", "run", "main.py"]
