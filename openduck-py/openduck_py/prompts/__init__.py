import os

from jinja2 import Template


def prompt(prompt_name: str, variables=None) -> str:
    file_path = os.path.join(os.path.dirname(__file__), prompt_name)
    with open(file_path, "r") as file:
        s = file.read()
    if variables is not None:
        t = Template(s)
        s = t.render(variables)
    return s
