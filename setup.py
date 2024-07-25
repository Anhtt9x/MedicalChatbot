from setuptools import setup , find_packages

setup(name="Medical-Chatbot",
      version="0.0.0",
      author="Anhtt9x",
      author_email="anhtt454598@gmail.com",
      packages=find_packages(where="./src", include=["*","src"]),
      )