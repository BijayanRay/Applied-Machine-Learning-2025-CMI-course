# Use an official Python runtime
FROM python:3.12

# Set the working directory in the container
WORKDIR /app
# Copy necessary files into the container
COPY requirements.txt .
COPY app.py .
COPY score.py .
COPY test_score.py .
COPY support_vector_machine_best_model.pkl .
COPY vectorizer.pkl .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
