## Setup & Execution

### Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop) installed and running

### 1. Build and Run

```bash
docker compose up -d --build && docker compose exec backtest bash
```

This will:
- Build the Docker image from the `Dockerfile` (based on the NautilusTrader image)
- Mount your current folder into the container
- Open a shell inside the container

### 2. Run Without Rebuilding

```bash
docker compose up -d && docker compose exec backtest bash
```

### 3. Run the Backtest

Inside the container shell:

```bash
python backtest_fx_bars.py
```

Results are saved to `output/` in your local project folder.

### 4. Stop the Container

```bash
docker compose down
```

