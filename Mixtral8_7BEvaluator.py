import os
import tiktoken
from LLMNeedleHaystackTester import LLMNeedleHaystackTester
from langchain_together import Together



class Mixtral8_7BEvaluator(LLMNeedleHaystackTester):
    def __init__(self,**kwargs):
        if 'together_api_key' not in  kwargs and not os.getenv('TOGETHER_API_KEY'):
            raise ValueError("Either together_api_key must be supplied with init, or TOGETHER_API_KEY must be in env")

        if 'model_name' not in kwargs:
            raise ValueError("model_name must be supplied with init, accepted model_names are ''")
        elif kwargs['model_name'] not in ['Mixtral-8x7B-Instruct-v0.1']:
            raise ValueError(f"Model name must be in this list (Mixtral-8x7B-Instruct-v0.1) but given {kwargs['model_name']}")

        if 'evaluation_method' not in kwargs:
            print("since evaluation method is not specified , 'gpt4' will be used for evaluation")
        elif kwargs['evaluation_method'] not in ('gpt4', 'substring_match'):
            raise ValueError("evaluation_method must be 'substring_match' or 'gpt4'")


        self.google_api_key = kwargs.pop('together_api_key', os.getenv('TOGETHER_API_KEY'))
        self.model_name = kwargs['model_name']
        self.model_to_test_description = kwargs.pop('model_name')
        self.tokenizer = tiktoken.encoding_for_model('gpt-4-1106-previe3') # Probably need to change in future
        self.model_to_test = Together(model=f'mistralai/{self.model_name}', temperature=0)
        kwargs['context_lengths_max'] = 32000
        super().__init__(**kwargs)



    def get_encoding(self,context):
        return self.tokenizer.encode(context)

    def get_decoding(self, encoded_context):
        return self.tokenizer.decode(encoded_context)

    def get_prompt(self, context):
        with open('Google_prompt.txt', 'r') as file: # Need to change it
            prompt = file.read()
        return prompt.format(retrieval_question=self.retrieval_question, context=context)

    async def get_response_from_model(self, prompt):
        response = self.model_to_test.invoke( # Need to make it async
            prompt
        )
        return response




if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    ht = Mixtral8_7BEvaluator(model_name='Mixtral-8x7B-Instruct-v0.1', evaluation_method='substring_match',
                         save_results_dir='results/Mixtral8_7B/results-run1',
                         save_contexts_dir='contexts/Mixtral8_7B/contexts-run1'
                         )

    ht.start_test()