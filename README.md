# Openduck

## Goals

Building an open source talking plush with a voice you can talk to.

Our kids should grow up in a household full of hackable droids they can play with, work on, and learn from.

<img src="goal.webp" width="450px"/>

## Setup

1. Copy the `.env.example` to `.env` and fill in real values.

2. Download server-side models and put them inside of `openduck-py/models` (**TODO: share models**.)

### Install Dependencies

1. Install espeak (`brew install espeak` on Mac OS or `sudo apt-get install espeak-ng` on Debian Linux).
   
   - There may be other environment variables you need to set on Mac OS. I had to: `export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib`.
     
3. `pip install -r openduck-py/requirements.txt`

### Without Docker

```
cd openduck-py
uvicorn openduck_py.routers.main:app --reload --env-file ../.env
```

### With Docker

`docker-compose up`

### Run the client

#### Simple Python Client

```
cd clients/simple
# Lighter-weight requirements
pip install -r requirements.txt
python simple_bot.py --record
```

#### Conversation Review
This will start up streamlit on port 8501 so make sure that port is forwarded if you are runnin on ssh. 

Install streamlit
```bash
pip install streamlit
```

Run streamlit
```bash
cd openduck-py
streamlit run observability.py
```

Alternative is to use [Dockerfile](Dockerfile.observability)

### Troubleshooting

- The Simple Python client sometimes has strange audio playback bugs. You can try restarting your OS's audio services, e.g. `sudo pkill coreaudiod` on Mac OS.
