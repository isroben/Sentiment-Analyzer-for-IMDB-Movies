# =========================
# BASE IMAGE
# =========================
FROM python:3.11-slim

# =========================
# SET WORKING DIRECTORY
# =========================
WORKDIR /app

# =========================
# COPY REQUIREMENTS
# =========================
COPY requirements.txt .

# =========================
# INSTALL DEPENDENCIES
# =========================
RUN pip install --no-cache-dir -r requirements.txt

# =========================
# COPY PROJECT FILES
# =========================
COPY . .

# =========================
# EXPOSE PORT
# =========================
EXPOSE 8000

# =========================
# RUN FASTAPI SERVER
# =========================
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]