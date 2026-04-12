## Setup & Execution

### Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop) installed and running

### 1. Build and Run

From the project directory, run:

```bash
docker compose up --build
```

This will:
- Build the Docker image from the `Dockerfile`
- Mount your current folder into the container



### 3. Cleanup

```bash
docker compose down
```

