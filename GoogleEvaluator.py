import os
import tiktoken
from LLMNeedleHaystackTester import LLMNeedleHaystackTester
import google.generativeai as genai
import time


class GoogleEvaluator(LLMNeedleHaystackTester):
    def __init__(self,**kwargs):
        if 'google_api_key' not in  kwargs and not os.getenv('GOOGLE_API_KEY'):
            raise ValueError("Either google_api_key must be supplied with init, or GOOGLE_API_KEY must be in env")

        if 'model_name' not in kwargs:
            raise ValueError("model_name must be supplied with init, accepted model_names are 'gemini-pro'")
        elif kwargs['model_name'] not in ['gemini-pro']:
            raise ValueError("Model name must be in this list (gemini-pro)")

        if 'evaluation_method' not in kwargs:
            print("since evaluation method is not specified , 'gpt4' will be used for evaluation")
        elif kwargs['evaluation_method'] not in ('gpt4', 'substring_match'):
            raise ValueError("evaluation_method must be 'substring_match' or 'gpt4'")


        self.google_api_key = kwargs.pop('google_api_key', os.getenv('GOOGLE_API_KEY'))
        self.model_name = kwargs['model_name']
        self.model_to_test_description = kwargs.pop('model_name')
        self.tokenizer = tiktoken.encoding_for_model('gpt-4-1106-previe3') # Since the tokenizer for Gemini is unknow, using gpt4 tokenizer

        genai.configure(api_key=self.google_api_key)
        self.model_to_test = genai.GenerativeModel(model_name=self.model_name)
        kwargs['context_lengths_max'] = 31000
        super().__init__(**kwargs)



    def get_encoding(self,context):
        return self.tokenizer.encode(context)

    def get_decoding(self, encoded_context):
        return self.tokenizer.decode(encoded_context)

    def get_prompt(self, context):
        with open('Google_prompt.txt', 'r') as file:
            prompt = file.read()
        return prompt.format(retrieval_question=self.retrieval_question, context=context)

    async def get_response_from_model(self, prompt):
        response = await self.model_to_test.generate_content_async(
            contents=prompt
        )
        return response.text




if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    ht = GoogleEvaluator(model_name='gemini-pro', evaluation_method='substring_match')

    ht.start_test()