if [ ! -d "/var/cache/venv" ]; then
    python3 -m venv /var/cache/venv
    source /var/cache/venv/bin/activate
    pip install -r requirements.txt
else
    source /var/cache/venv/bin/activate
fi