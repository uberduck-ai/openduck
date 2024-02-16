from openduck_py.voices import styletts2


styletts2.styletts2_inference(
    text="Hello, my name is Matthew. How are you today?",
    model_path="styletts2/rap_v1.pt",
    model_bucket="uberduck-models-us-west-2",
    config_path="styletts2/rap_v1_config.yml",
    config_bucket="uberduck-models-us-west-2",
    output_bucket="uberduck-audio-outputs",
    output_path="test.wav",
    style_prompt_path="511f17d1-8a30-4be8-86aa-4cdd8b0aed70.wav",
    style_prompt_bucket="uberduck-audio-files",
)
