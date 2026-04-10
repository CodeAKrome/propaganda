#!/usr/bin/env python3

"""
Large Language Model integration and processing module.
"""
"""
Module: geminiai_listmodels.py
Purpose: Module for geminiai_listmodels.py functionality.
Usage: python ./llm/geminiai_listmodels.py [arguments]
"""
#!/usr/bin/env python

import os
import google.generativeai as genai
#import google.genai as genai

# ./geminiai_listmodels.py|./geminimodelfilter.pl>gemtest_runall.sh

# Configure the API key (replace with your actual API key or set as environment variable)
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE"))


def list_models():
    try:
        # List all available models
        models = genai.list_models()

        # Print model details
        for model in models:
            print(f"Model Name: {model.name}")
            print(f"Description: {model.description}")
            print(f"Supported Methods: {', '.join(model.supported_generation_methods)}")
            print("-" * 50)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    list_models()
