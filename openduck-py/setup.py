from setuptools import find_packages, setup

# TODO (Sam): update.
setup(
    name="openduck-py",
    version="0.0.1",
    author="Zach Ocean, Sam Koelle, William Luer, Matthew Kennedy",
    author_email="z@uberduck.ai",
    description="Opensource API for multimedia AI",
    url="https://github.com/uberduck-ai/openduck",
    packages=find_packages(),
    long_description="Uberduck python",
    long_description_content_type="text/markdown",
    classifiers=["Development Status :: 3 - Alpha", "Topic :: Utilities"],
    include_package_data=True,
)
