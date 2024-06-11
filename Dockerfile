# Use the official Python image from the Docker Hub
# can also use smaller python image (slim)
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Set environment variables
ENV HF_TOKEN=your_hugging_face_token

# Run the fine-tuning script
# CMD ["python", "main_generate_labels.py"]
CMD ["python", "main_finetune_model.py"]
