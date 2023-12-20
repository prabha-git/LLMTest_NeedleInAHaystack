import numpy as np
import os
import json
import time
from datetime import datetime, timezone
import glob
import asyncio
from asyncio import Semaphore


from langchain.chat_models import ChatOpenAI

import google.generativeai as genai


from langchain.evaluation import load_evaluator

from dotenv import load_dotenv
load_dotenv()
class GLLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """

    def __init__(self,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                 haystack_dir="PaulGrahamEssays",
                 retrieval_question="What is the best thing to do in San Francisco?",
                 results_version=1,
                 context_lengths_min=1000,
                 context_lengths_max=140000,
                 context_lengths_num_intervals=35,
                 context_lengths=None,
                 document_depth_percent_min=0,
                 document_depth_percent_max=100,
                 document_depth_percent_intervals=35,
                 document_depth_percents=None,
                 document_depth_percent_interval_type="linear",
                 model_provider="Google",
                 openai_api_key=None,
                 google_api_key=None,
                 model_name='gemini-pro',
                 num_concurrent_requests=1,
                 save_results=True,
                 save_contexts=True,
                 final_context_length_buffer=2000,
                 seconds_to_sleep_between_completions=None,
                 print_ongoing_status=True):
        """
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_provider: The provider of the model. Must be either 'OpenAI' or 'Anthropic'. Default is 'OpenAI'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")

        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError(
                    "Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(
                    np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals,
                                endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError(
                    "Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(
                        np.linspace(document_depth_percent_min, document_depth_percent_max,
                                    num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in
                                                    np.linspace(document_depth_percent_min, document_depth_percent_max,
                                                                document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError(
                "document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")

        if model_provider not in ["Google"]:
            raise ValueError("model_provider must be 'Google'")

        if model_provider == "Google" and   model_name not in ["gemini-pro"]:
            raise ValueError(
                "If the model provider is 'Google', the model name must be gemini-pro")

        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.model_name = model_name

        # if not self.openai_api_key and not os.getenv('OPENAI_API_KEY'):
        #     raise ValueError(
        #         "Either openai_api_key must be supplied with init, or OPENAI_API_KEY must be in env. Used for evaluation model")
        # else:
        #     self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')

        # Google API KEY

        self.google_api_key = google_api_key or os.getenv('GOOGLE_API_KEY')

        if self.model_provider == "Google":
            if not self.google_api_key and not os.getenv('GOOGLE_API_KEY'):
                raise ValueError(
                    "Either google_api_key  must be supplied with init, or GOOGLE_API_KEY must be in env.")
            else:
                self.google_api_key = google_api_key or os.getenv('GOOGLE_API_KEY')

        if not self.model_name:
            raise ValueError("model_name must be provided.")

        if model_provider == "Google":
            genai.configure(api_key=self.google_api_key)
            self.model_to_test = genai.GenerativeModel(model_name=self.model_name)

        self.model_to_test_description = model_name
        #self.evaluation_model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0, openai_api_key=self.openai_api_key)

    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = 'results/'
        if not os.path.exists(results_dir):
            return False

        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    def get_context_length_in_chars(self, context):
        return len(context)

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_chars(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def encode_and_trim(self, context, context_length):
        return context[:context_length]

    def insert_needle(self, context, depth_percent, context_length):
        new_context=""
        if len(context) + len(self.needle) > context_length:
            context = context[:context_length - len(self.needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            new_context = context + self.needle

        elif depth_percent == 0:
            new_context =  self.needle + context

        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            new_context = context[:insertion_point]

            # Then we iteration backwards until we find the first period
            while new_context and new_context[-1] not in '.':
                insertion_point -= 1
                new_context = context[:insertion_point]

            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            new_context += self.needle + context[insertion_point:]


        return new_context


    def generate_prompt(self, context):
        if self.model_provider == "Google":
            with open('Google_prompt.txt', 'r') as file:
                prompt = file.read()
            return prompt.format(retrieval_question=self.retrieval_question, context=context)




    async def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context


    def evaluate_response(self, response):
        accuracy_criteria = {
            "accuracy": """
            Score 1: The answer is completely unrelated to the reference.
            Score 3: The answer has minor relevance but does not align with the reference.
            Score 5: The answer has moderate relevance but contains inaccuracies.
            Score 7: The answer aligns with the reference but has minor omissions.
            Score 10: The answer is completely accurate and aligns perfectly with the reference.
            Only respond with a numberical score
            """
        }

        # Using GPT-4 to evaluate
        evaluator = load_evaluator(
            "labeled_score_string",
            criteria=accuracy_criteria,
            llm=self.evaluation_model,
        )

        eval_result = evaluator.evaluate_strings(
            # The models response
            prediction=response,

            # The actual answer
            reference=self.needle,

            # The question asked
            input=self.retrieval_question,
        )

        return int(eval_result['score'])

    def evaluate_response_basic(self,response):
        if 'sandwich' in response.lower() and 'dolores' in response.lower():
            return 1
        else:
            return 0

    async def evaluate_and_log(self, context_length, depth_percent):

        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                return

        # Go generate the required length context and place your needle statement in
        context = await self.generate_context(context_length, depth_percent)

        # Prepare your message to send to the model you're going to evaluate
        prompt = self.generate_prompt(context)

        test_start_time = time.time()

        # Go see if the model can answer the question to pull out your random fact
        if self.model_provider == "Google":
            print(f"\nChecking the context length in tokens: {self.model_to_test.count_tokens(prompt)}\n")
            response = await self.model_to_test.generate_content_async(prompt)
            response = response.text


        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        # Compare the reponse to the actual needle you placed
        #score = self.evaluate_response(response)

        score = self.evaluate_response_basic(response)

        results = {
            # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
            'model': self.model_to_test_description,
            'context_length': int(context_length),
            'depth_percent': float(depth_percent),
            'version': self.results_version,
            'needle': self.needle,
            'model_response': response,
            'score': score,
            'test_duration_seconds': test_elapsed_time,
            'test_timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print(f"-- Test Summary -- ")
            print(f"Duration: {test_elapsed_time:.1f} seconds")
            print(f"Context: {context_length} tokens")
            print(f"Depth: {depth_percent}%")
            print(f"Score: {score}")
            print(f"Response: {response}\n")

        context_file_location = f'{self.model_name.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent * 100)}'

        if self.save_contexts:
            results['file_name']: context_file_location

            # Save the context to file for retesting
            if not os.path.exists('contexts'):
                os.makedirs('contexts')

            with open(f'contexts/{context_file_location}_context.txt', 'w') as f:
                f.write(context)

        if self.save_results:
            # Save the context to file for retesting
            if not os.path.exists('results'):
                os.makedirs('results')

            # Save the result to file for retesting
            with open(f'results/{context_file_location}_results.json', 'w') as f:
                json.dump(results, f)

        if self.seconds_to_sleep_between_completions:
            await asyncio.sleep(self.seconds_to_sleep_between_completions)

    async def bound_evaluate_and_log(self, sem, *args):
        async with sem:
            await self.evaluate_and_log(*args)

    async def run_test(self):
        sem = Semaphore(self.num_concurrent_requests)

        # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(sem, context_length, depth_percent)
                tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        asyncio.run(self.run_test())

if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    ht = GLLMNeedleHaystackTester()

    ht.start_test()