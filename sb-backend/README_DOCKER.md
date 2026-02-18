# SoulBuddy Backend Dockerization

This guide explains how to build and run the SoulBuddy backend using Docker.

## Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed
- (Optional) [Docker Compose](https://docs.docker.com/compose/) if using `docker-compose.yml`

## Building the Docker Image

From the `sb-backend` directory, run:

```
docker build -t sb-backend .
```

This command builds the Docker image using the provided `Dockerfile` and tags it as `sb-backend`.

## Running the Container

To run the backend server container and expose it on port 8000:

```
docker run -p 8000:8000 --env-file .env.demo sb-backend
```

- `-p 8000:8000` maps the container's port 8000 to your host's port 8000.
- `--env-file .env.demo` loads environment variables from `.env.demo`.


## Using Docker Compose

You can run the backend using Docker Compose with the provided `docker-compose.yml` file. This is especially useful for local development and testing.

### Example `docker-compose.yml`

```
version: '3.9'

services:
	sb-backend:
		build: ./sb-backend
		container_name: sb-backend
		environment:
			# Set your environment variables here
			# DATABASE_URL: postgresql://user:password@host:port/dbname
			# DATA_DB_URL: postgresql://user:password@host:port/dbname
			# Add other env vars as needed, e.g. OPENAI_API_KEY
		ports:
			- "8000:8000"
		volumes:
			- ./sb-backend:/app
```

### Running the Backend

From the project root, run:

```
docker compose up --build
```

This will build and start the `sb-backend` service. The backend will be available at `http://localhost:8000`.

You can stop the service with `Ctrl+C` or by running:

```
docker compose down
```

## Customizing Environment Variables

- The backend uses environment variables for configuration.
- Edit `.env.demo` or provide your own `.env` file as needed.

## Development Tips
- Use a `.dockerignore` file to avoid copying unnecessary files into the image.
- The Dockerfile installs dependencies first for efficient caching.
- The entrypoint script (`entrypoint.sh`) is used to start the server.

## Troubleshooting
- Ensure ports are not already in use on your host.
- Check logs with `docker logs <container_id>` if the server does not start.
- If you change dependencies, rebuild the image with `docker build -t sb-backend .`

---

For more details, see the main project README or contact the maintainers.
