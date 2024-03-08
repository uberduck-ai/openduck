import os


def prompt(prompt_name: str, template_vars: dict = {}):
    file_path = os.path.join(os.path.dirname(__file__), f"{prompt_name}.md")
    with open(file_path, "r") as file:
        prompt = file.read()
