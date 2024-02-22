# Openduck

## Goals

Make it easy to build interactive, open, multimodal AI software and hardware products.

Our kids should grow up in a household full of hackable droids they can play with, work on, and learn from.

<img src="goal.webp" width="450px"/>

## Setup

1. Copy the `.env.example` to `.env` and fill in real values.

2. Download server-side models and put them inside of `uberduck-py/models` (**TODO: share models**.)

### Without Docker

`uvicorn openduck_py.routers.main:app --reload --env-file .env`

### With Docker

`docker-compose up`
