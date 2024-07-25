import json
import yaml
import requests
import backoff
from openai import AzureOpenAI
import http.client
import ollama

class ContentFormatter:
    @staticmethod
    def chat_completions(text, settings_params):
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ]
        data = {"messages": message, **settings_params}
        return json.dumps(data)

class AzureAgent:
    def __init__(self, model_name):
        with open("settings.yaml", "r") as stream:
            try:
                model_settings = yaml.safe_load(stream)[model_name]
            except yaml.YAMLError as exc:
                print(exc)
                return

        self.azure_uri = model_settings['AZURE_ENDPOINT_URL']
        self.headers = {
            'Authorization': f"Bearer {model_settings['AZURE_ENDPOINT_API_KEY']}",
            'Content-Type': 'application/json'
        }
        self.chat_formatter = ContentFormatter

    def invoke(self, text, **kwargs):
        body = self.chat_formatter.chat_completions(text, {**kwargs})
        conn = http.client.HTTPSConnection(self.azure_uri)
        conn.request("POST", '/v1/chat/completions', body=body, headers=self.headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        decoded_data = data.decode("utf-8")
        parsed_data = json.loads(decoded_data)
        content = parsed_data["choices"][0]["message"]["content"]
        return content

class GPTAgent:
    def __init__(self, model_name):
        with open("settings.yaml", "r") as stream:
            try:
                model_settings = yaml.safe_load(stream)[model_name]
            except yaml.YAMLError as exc:
                print(exc)
                return

        self.client = AzureOpenAI(
            api_key=model_settings['AZURE_OPENAI_KEY'],
            api_version=model_settings['AZURE_OPENAI_VERSION'],
            azure_endpoint=model_settings['AZURE_OPENAI_ENDPOINT']
        )
        self.deployment_name = model_settings['AZURE_DEPLOYMENT_NAME']

    def invoke(self, text, **kwargs):
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text}
            ],
            **kwargs
        )
        return response.choices[0].message.content

class OllamaModel:
    def __init__(self, model_name = 'llama3o', system_prompt = 'You are a helpful assistant'):
        self.model_name = model_name
        self.model_create(model_name, system_prompt)

    def model_create(self, model_name, system_prompt):
        modelfile = f'FROM llama3\nSYSTEM {system_prompt} \n'
        print(modelfile)
        ollama.create(model=model_name, modelfile=modelfile)

    def invoke(self, prompt, mode = 'normal'):
        if mode == 'bin_classification':
            error = 0
            while True:
                answer = ollama.generate(model=self.model_name, prompt=prompt)
                print(answer['response'])
                if answer['response'].lower().startswith('yes'):
                    return 1
                elif answer['response'].lower().startswith('no'):
                    return 0
                else:
                    error += 1
                if error > 10:
                    return -1
        else:
            answer = ollama.generate(model=self.model_name, prompt=prompt)
            return answer['response']

if __name__ == '__main__':
    agents = {
        'GPT4-turbo': GPTAgent,
        'GPT35-turbo': GPTAgent,
        'Llama2-7B-chat': AzureAgent,
        'Llama2-13B-chat': AzureAgent,
        'Llama2-70B-chat': AzureAgent,
        'Mistral-large': AzureAgent
    }

    for model_name, Agent in agents.items():
        if model_name.startswith('GPT3'):
            agent = Agent(model_name)
        # agent = Agent(model_name)
            print(agent.invoke("Hello who are you?", temperature=0, max_tokens=5))