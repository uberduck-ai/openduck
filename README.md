# Openduck

## Goals

Make it easy to build interactive, open, multimodal AI software and hardware products.

Our kids should grow up in a household full of hackable droids they can play with, work on, and learn from.

<img src="goal.webp" width="250px"/>

## Setup

1. Copy the `.env.example` to `.env` and fill in real values.

### Without Docker

`uvicorn openduck_py.routers.main:app --reload --env-file .env`

### With Docker

`docker-compose up`

## Proofs of Concept

A working list of applications that will be simple to build using this repo as a starting point.

- [ ] AI Podcast Content Assistant—an AI that interviews you, then creates shareable short form video clips from the recording.

- [ ] Interactive AI Plushy—a physical plushy you can talk to with your voice.