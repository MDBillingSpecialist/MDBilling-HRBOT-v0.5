import sys
import os
import unittest
from dotenv import load_dotenv
from openai import AzureOpenAI

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.config_manager import config

class TestAzureOpenAIConnection(unittest.TestCase):
    def setUp(self):
        load_dotenv()
        self.client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version="2023-05-15",
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT
        )

    def test_azure_openai_connection(self):
        try:
            response = self.client.chat.completions.create(
                model=config.AZURE_OPENAI_DEPLOYMENT_NAME,
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
        self.assertIsNotNone(config.AZURE_OPENAI_API_KEY)
        self.assertIsNotNone(config.AZURE_OPENAI_ENDPOINT)
        self.assertIsNotNone(config.AZURE_OPENAI_DEPLOYMENT_NAME)
        print(f"API Key: {config.AZURE_OPENAI_API_KEY[:5]}...")
        print(f"Endpoint: {config.AZURE_OPENAI_ENDPOINT}")
        print(f"Deployment Name: {config.AZURE_OPENAI_DEPLOYMENT_NAME}")

if __name__ == '__main__':
    unittest.main()