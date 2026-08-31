FROM python:3.12-slim

# Non-root user: never run trading code as root
RUN useradd --create-home trader
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY algotrader/ algotrader/
COPY run.py .
COPY run_forex.py .
COPY tests/ tests/

# SAFETY: paper mode baked in. Live mode requires overriding BOTH vars
# at `docker run` time — the image itself can never trade real money.
ENV ALGO_TRADING_MODE=paper
ENV PYTHONUNBUFFERED=1

USER trader

# container healthcheck = test suite must pass at build time
RUN python -m pytest tests/ -q

ENTRYPOINT ["python", "run.py"]
CMD ["paper", "--symbols", "AAPL", "MSFT", "--offline"]
