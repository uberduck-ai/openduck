import os


def prompt(prompt_name: str):
    file_path = os.path.join(os.path.dirname(__file__), f"{prompt_name}.md")
    with open(file_path, "r") as file:
        return file.read()
