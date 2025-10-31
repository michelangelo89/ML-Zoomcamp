# Start from the prepared base (includes pipeline_v2.bin)
FROM agrigorev/zoomcamp-model:2025

# Bring in uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code
ENV PATH="/code/.venv/bin:${PATH}"

# Copy project dependency info
COPY pyproject.toml uv.lock ./
RUN uv sync --locked

# Copy app file
COPY predict.py ./

EXPOSE 9696
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]