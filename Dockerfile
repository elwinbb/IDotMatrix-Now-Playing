ARG BUILD_FROM
FROM ${BUILD_FROM}

# Install system dependencies for Bluetooth LE (BlueZ) and Python packages
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-numpy \
    bluez \
    bluez-libs \
    dbus \
    jq

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application
COPY now_playing.py .

# Copy startup script
COPY run.sh /run.sh
RUN chmod a+x /run.sh

CMD ["/run.sh"]
