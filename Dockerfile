# Base image with Python
FROM python:3.10.14-slim

# Install Poetry for dependency management
RUN pip install --no-cache-dir poetry==1.8.3

# Set environment variables for Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Set the working directory
WORKDIR /app

# Copy the Turing test folder into the container
COPY . /app/turing

# Set the working directory to turing folder
WORKDIR /app/turing

# Install the Python dependencies from pyproject.toml
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

# Expose the port that your application will run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
