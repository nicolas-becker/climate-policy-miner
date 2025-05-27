# Use a lightweight Conda base image
FROM continuumio/miniconda3:latest

# Set the working directory in the container
WORKDIR /app

# Copy the application code and environment file into the container
COPY . /app

# Install system-level dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 poppler-utils

# Install dependencies from the Conda environment file
RUN conda env create -f environment_app.yml

# Activate the environment and ensure it is used by default
RUN echo "conda activate transportpolicyminer_venv" >> ~/.bashrc
ENV PATH=/opt/conda/envs/transportpolicyminer_venv/bin:$PATH

# Expose the port Flask will run on
EXPOSE 5000

# Set the environment variable for Flask
ENV FLASK_APP=src/flask_app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the Flask app
#CMD ["flask", "run"] # Uncomment this line to run with Flask's development server
# Use Gunicorn for production deployment
# -w 4: 4 worker processes
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "src.flask_app:app"]