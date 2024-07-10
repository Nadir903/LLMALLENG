import argparse
import json
from os.path import join as juntador
from transformers import AutoTokenizer as t
from transformers import AutoModelForSeq2SeqLM as m

mi_modelo = "fine_tuned_model"
mi_tokenizador = t.from_pretrained(mi_modelo)
mi_modelo = m.from_pretrained(mi_modelo)


def handle_user_query(query, query_id, output_path):
    inputs = mi_tokenizador.encode(query, return_tensors="pt", truncation=True)
    outputs = mi_modelo.generate(inputs)
    translation = mi_tokenizador.decode(outputs[0], skip_special_tokens=True)

    result = {
        "generated_query": translation,
        "detected_language": "en",
    }

    with open(juntador(output_path, f"{query_id}.json"), "w") as f:
        json.dump(result, f)


parsero_Parser = argparse.ArgumentParser(description='Run the inference.')
parsero_Parser.add_argument('--query', type=str, help='The user query.', required=True, action="append")
parsero_Parser.add_argument('--query_id', type=str, help='The IDs for the queries, in the same order as the queries.',
                    required=True, action="append")
parsero_Parser.add_argument('--output', type=str, help='Path to the output directory.', required=True)

if __name__ == "__main__":
    args = parsero_Parser.parse_args()
    queries = args.query
    query_ids = args.query_id
    output = args.output

    assert len(queries) == len(query_ids), "The number of queries and query IDs must be the same."

    for busqueda, identificacion in zip(queries, query_ids):
        handle_user_query(busqueda, identificacion, output)
