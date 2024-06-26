# This file will be executed when a user wants to query your project.
import argparse
from os.path import join
import json
import requests as r
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Done Implement the inference logic here

# Load the tokenizer and model :pip install sentencepiece
model_name = "Helsinki-NLP/opus-mt-mul-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# Function to handle user query
def handle_user_query(query, query_id, output_path):
    inputs = tokenizer.encode(query, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result = {
        "generated_queries": [translation],
        "detected_language": "en",
    }

    with open(join(output_path, f"{query_id}.json"), "w") as f:
        json.dump(result, f)


# TODO OPTIONAL
# This function is optional for you
# You can use it to interfer with the default ranking of your system.
#
# If you do embeddings, this function will simply compute the cosine-similarity
# and return the ordering and scores
def rank_articles(generated_queries, article_representations):
    """
    This function takes as arguments the generated / augmented user query, as well as the
    transformed article representations.
    
    It needs to return a list of shape (M, 2), where M <= #article_representations.
    Each tuple contains [index, score], where index is the index in the article_repr array.
    The list need already be ordered by score. Higher is better, between 0 and 1.
    
    An empty return list indicates no matches.
    """

    system_prompt = """You are a helpful assistant working at the EU. It is your job to give users unbiased article recommendations. You will be given a user query, and a list of articles. Your task is to rank the articles by relevance to the query. Response with a list of article indices and rankings as nested JSON arrays. The first index should be the most relevant article, the last the least relevant. The indices should be 0-based. The rankings should be "VERY", "HIGH", "MEDIUM", or "IRRELEVANT". If there are multiple queries, it is only variations of one original one. Only return the rankings as valid JSON, for example [[ 3, "HIGH" ], [5, "MEDIUM"]]. Do not give me anything else, nor any explanation."""

    article_string = [f"Article {index}: {json.dumps(article)}" for index, article in
                      enumerate(article_representations)]
    user_prompt = "User Query: " + json.dumps(generated_queries) + "\n\n" + "Articles: " + "\n".join(article_string)

    # This call is using the raw mode (raw=True). This mode gives you more flexibility in the response, but is a bit more complex.
    # You need to look up the template file of your model and format the request accordingly.
    # If you use raw, other parameters, in particular system, have no effect, as raw overrides them.
    #
    # The key advantage of the raw mode is that you can force the model to start with a specific string.
    # In the example below, we start the model response with "[[", which strongly conditions the model to output JSON.
    call_data = {
        "model": "llama3",
        "stream": False,
        "raw": True,
        "prompt": f"""<|start_header_id|>system<|end_header_id|>
        
        {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
        
        {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>[["""
    }

    response = r.post("http://localhost:11434/api/generate", json=call_data)
    response = response.json()

    # The model will not regenerate "[[", so we need to add it back in.
    response = json.loads('[[' + response["response"])

    def s2i(score):
        if score == "VERY": return 1.0
        if score == "HIGH": return 0.75
        if score == "MEDIUM": return 0.5
        return 0.0

    response = [[index, s2i(score)] for index, score in response]
    print(response)


if False:
    # handle_user_query("What are the benefits of LLMs in programming?", "1", "output")
    rank_articles(["What are the benefits of LLMs in programming?"],
                  [["llms", "ai", "programming"], ["war in ukraine", "russia", "ukraine"]])
    exit(0)

#exit(0)

# This is a sample argparse-setup, you probably want to use in your project:
parser = argparse.ArgumentParser(description='Run the inference.')
parser.add_argument('--query', type=str, help='The user query.', required=True, action="append")
parser.add_argument('--query_id', type=str, help='The IDs for the queries, in the same order as the queries.',
                    required=True, action="append")
parser.add_argument('--output', type=str, help='Path to the output directory.', required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    queries = args.query
    query_ids = args.query_id
    output = args.output

    assert len(queries) == len(query_ids), "The number of queries and query IDs must be the same."

    for query, query_id in zip(queries, query_ids):
        handle_user_query(query, query_id, output)

#python user_inference.py --query "Translate this text" --query_id 1 --output /Users/nadiralcalde/Seminarios/Generative_AI_and_Democracy/LLMALLENG/output
