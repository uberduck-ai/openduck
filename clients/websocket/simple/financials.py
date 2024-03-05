# COGS

# Inputs
production_volume = 500
prompt_tokens_per_request = 500  # Server logs print the number of prompt and completion tokens used on each request
completion_tokens_per_request = 30
# Token prices from https://openai.com/pricing for gpt-3.5-turbo-0125
prompt_token_price = 0.0000005  # $/token
completion_token_price = 0.0000015  # $/token

# Runtimes from profiling on a g4dn.xlarge, 3 sentences of response
stt_runtime = 0.15  # seconds
tts_runtime = 2.0  # seconds per inference
ec2_price = 0.526  # $/hour, g4dn.xlarge ec2 instance
usage_minutes_per_day = 60  # Minutes per user per day
requests_per_minute = 3  # How many responses are generated per minute for each user

# Calculation

plushy_cost = 12.50
if 1000 <= production_volume < 5000:
    plushy_cost = 9.00
if production_volume >= 5000:
    plushy_cost = 6.00

# Requests per day from 1 user
requests_per_day = usage_minutes_per_day * requests_per_minute
tts_cost_per_request = tts_runtime * ec2_price / 3600  # seconds * ($ / second)
stt_cost_per_request = stt_runtime * ec2_price / 3600
llm_cost_per_request = (
    prompt_tokens_per_request * prompt_token_price
    + completion_tokens_per_request * completion_token_price
)
requests_per_year = requests_per_day * 365

llm_cost_annual = llm_cost_per_request * requests_per_year
stt_cost_annual = stt_cost_per_request * requests_per_year
tts_cost_annual = tts_cost_per_request * requests_per_year


print(f"Plushy cost with {usage_minutes_per_day} minutes of usage per day:")
print(f"Upfront: ${plushy_cost:.2f}")
print("Annual:")
print(f"LLM cost: ${llm_cost_annual:.2f} / year")
print(f"TTS cost: ${tts_cost_annual:.2f} / year")
print(f"STT cost: ${stt_cost_annual:.2f} / year")
print(f"Total: ${llm_cost_annual + tts_cost_annual + stt_cost_annual:.2f} / year")
