import sys
import os
import unittest
from dotenv import load_dotenv
from openai import OpenAI

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.config_manager import config

class TestOpenAIConnection(unittest.TestCase):
    def setUp(self):
        load_dotenv()
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def test_openai_connection(self):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",  # or whichever model you're using
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, are you working?"}
                ],
                max_tokens=50
            )
            self.assertIsNotNone(response)
            self.assertIsNotNone(response.choices[0].message.content)
            print(f"API Response: {response.choices[0].message.content}")
        except Exception as e:
            self.fail(f"API call failed: {str(e)}")

    def test_config_values(self):
        self.assertIsNotNone(config.OPENAI_API_KEY)
        print(f"API Key: {config.OPENAI_API_KEY[:5]}...")

if __name__ == '__main__':
    unittest.main()