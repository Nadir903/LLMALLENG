import argparse
import subprocess
import sys
from random import sample
from os import mkdir, rmdir
from os.path import join, isfile, isdir, split as split_path
import shutil
import json


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def nanoid(n=10):
    return ''.join(sample('abcdefghijklmnopqrstuvwxyz', n))


def tranform_output(output):
    lines = output.split('\n')
    lines = map(lambda x: x.strip(), lines)
    lines = filter(lambda x: x != '', lines)
    return "◦◦◦ " + "\n◦◦◦ ".join(lines)


def test_setup():
    result = subprocess.run([sys.executable, "user_setup.py"], check=False, capture_output=True)
    if result.returncode != 0:
        print(bcolors.FAIL + "> The setup script did not run successfully.")
        print(bcolors.FAIL + tranform_output(result.stderr.decode()))

    else:
        print(bcolors.OKGREEN + "> The setup script ran successfully.")
        print(bcolors.OKBLUE + tranform_output(result.stdout.decode()))


def test_preprocess():
    output_dir = nanoid()
    mkdir(output_dir)

    args = ["--output", output_dir]
    articles = ["sample_data/article_1.json"]

    for article in articles:
        args.append("--input")
        args.append(article)

    result = subprocess.run([sys.executable, "user_preprocess.py", *args], check=False, capture_output=True)
    if result.returncode != 0:
        print(bcolors.FAIL + "> The preprocess script did not run successfully.")
        print(bcolors.FAIL + tranform_output(result.stderr.decode()))

    else:
        try:
            for article in articles:
                assert isfile(
                    join(output_dir, split_path(article)[-1])), f"The file {split_path(article)[-1]} was not created."
                with open(join(output_dir, split_path(article)[-1]), "r") as f:
                    data = json.load(f)
                    assert "transformed_representation" in data, f"The key 'transformed_representation' was not found in the file {split_path(article)[-1]}."

            print(bcolors.OKGREEN + "> The preprocess script ran successfully.")
            print(bcolors.OKBLUE + tranform_output(result.stdout.decode()))
        except Exception as e:
            print(bcolors.FAIL + "> The preprocess script did not create the expected output.")
            print(bcolors.FAIL + tranform_output(str(e)))

    shutil.rmtree(output_dir, ignore_errors=True)


def test_prepare_dataset():
    output_dir = nanoid()
    mkdir(output_dir)

    # First, run preprocess to generate required files
    preprocess_args = ["--output", output_dir]
    articles = ["sample_data/article_1.json"]

    for article in articles:
        preprocess_args.append("--input")
        preprocess_args.append(article)

    preprocess_result = subprocess.run([sys.executable, "user_preprocess.py", *preprocess_args], check=False,
                                       capture_output=True)
    if preprocess_result.returncode != 0:
        print(bcolors.FAIL + "> The preprocess script did not run successfully.")
        print(bcolors.FAIL + tranform_output(preprocess_result.stderr.decode()))
        shutil.rmtree(output_dir, ignore_errors=True)
        return

    # Now, run prepare_dataset_not_tokenized
    result = subprocess.run([sys.executable, "prepare_dataset_not_tokenized.py"], check=False, capture_output=True)
    if result.returncode != 0:
        print(bcolors.FAIL + "> The prepare_dataset_not_tokenized script did not run successfully.")
        print(bcolors.FAIL + tranform_output(result.stderr.decode()))
    else:
        try:
            assert isdir("not_tokenized_dataset"), "The directory not_tokenized_dataset was not created."
            print(bcolors.OKGREEN + "> The prepare_dataset_not_tokenized script ran successfully.")
            print(bcolors.OKBLUE + tranform_output(result.stdout.decode()))
        except Exception as e:
            print(bcolors.FAIL + "> The prepare_dataset_not_tokenized script did not create the expected output.")
            print(bcolors.FAIL + tranform_output(str(e)))

    shutil.rmtree(output_dir, ignore_errors=True)


def test_fine_tune():
    output_dir = nanoid()
    mkdir(output_dir)

    # First, run prepare_dataset_not_tokenized to generate required files
    prepare_result = subprocess.run([sys.executable, "prepare_dataset_not_tokenized.py"], check=False,
                                    capture_output=True)
    if prepare_result.returncode != 0:
        print(bcolors.FAIL + "> The prepare_dataset_not_tokenized script did not run successfully.")
        print(bcolors.FAIL + tranform_output(prepare_result.stderr.decode()))
        shutil.rmtree(output_dir, ignore_errors=True)
        return

    # Now, run fine_tune_model
    result = subprocess.run([sys.executable, "fine_tune_model.py"], check=False, capture_output=True)
    if result.returncode != 0:
        print(bcolors.FAIL + "> The fine_tune_model script did not run successfully.")
        print(bcolors.FAIL + tranform_output(result.stderr.decode()))
    else:
        try:
            assert isdir("fine_tuned_model"), "The directory fine_tuned_model was not created."
            print(bcolors.OKGREEN + "> The fine_tune_model script ran successfully.")
            print(bcolors.OKBLUE + tranform_output(result.stdout.decode()))
        except Exception as e:
            print(bcolors.FAIL + "> The fine_tune_model script did not create the expected output.")
            print(bcolors.FAIL + tranform_output(str(e)))

    shutil.rmtree(output_dir, ignore_errors=True)


def test_inference():
    out_dir = nanoid()
    mkdir(out_dir)

    args = ["--output", out_dir]
    queries = ["sports", "soccer", "Munich vs Dortmund"]
    for query in queries:
        args.append("--query")
        args.append(query)

    query_ids = [nanoid() for _ in queries]
    for query_id in query_ids:
        args.append("--query_id")
        args.append(query_id)

    result = subprocess.run([sys.executable, "user_inference.py", *args], check=False, capture_output=True)
    if result.returncode != 0:
        print(bcolors.FAIL + "> The inference script did not run successfully.")
        print(bcolors.FAIL + tranform_output(result.stderr.decode()))

    else:
        try:
            for query_id in query_ids:
                assert isfile(join(out_dir, f"{query_id}.json")), f"The file {query_id}.json was not created."
                with open(join(out_dir, f"{query_id}.json"), "r") as f:
                    filedata = json.load(f)
                    assert "detected_language" in filedata, f"The key 'detected_language' was not found in the file {query_id}.json."
                    assert "generated_query" in filedata, f"The key 'generated_query' was not found in the file {query_id}.json."

            print(bcolors.OKGREEN + "> The inference script ran successfully.")
            print(bcolors.OKBLUE + tranform_output(result.stdout.decode()))

        except Exception as e:
            print(bcolors.FAIL + "> The inference script did not create the expected output.")
            print(bcolors.FAIL + tranform_output(str(e)))

    shutil.rmtree(out_dir, ignore_errors=True)


def test_evaluate():
    output_dir = nanoid()
    mkdir(output_dir)

    # First, run prepare_dataset_not_tokenized and fine_tune_model to generate required files
    prepare_result = subprocess.run([sys.executable, "prepare_dataset_not_tokenized.py"], check=False,
                                    capture_output=True)
    if prepare_result.returncode != 0:
        print(bcolors.FAIL + "> The prepare_dataset_not_tokenized script did not run successfully.")
        print(bcolors.FAIL + tranform_output(prepare_result.stderr.decode()))
        shutil.rmtree(output_dir, ignore_errors=True)
        return

    fine_tune_result = subprocess.run([sys.executable, "fine_tune_model.py"], check=False, capture_output=True)
    if fine_tune_result.returncode != 0:
        print(bcolors.FAIL + "> The fine_tune_model script did not run successfully.")
        print(bcolors.FAIL + tranform_output(fine_tune_result.stderr.decode()))
        shutil.rmtree(output_dir, ignore_errors=True)
        return

    # Now, run evaluate_model
    result = subprocess.run([sys.executable, "evaluate_model.py"], check=False, capture_output=True)
    if result.returncode != 0:
        print(bcolors.FAIL + "> The evaluate_model script did not run successfully.")
        print(bcolors.FAIL + tranform_output(result.stderr.decode()))
    else:
        try:
            print(bcolors.OKGREEN + "> The evaluate_model script ran successfully.")
            print(bcolors.OKBLUE + tranform_output(result.stdout.decode()))
        except Exception as e:
            print(bcolors.FAIL + "> The evaluate_model script did not produce the expected output.")
            print(bcolors.FAIL + tranform_output(str(e)))

    shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess the data.')
    parser.add_argument('--part', type=str, help='Which part of the pipeline to test.', required=True,
                        choices=['preprocess', 'setup', 'inference', 'prepare_dataset', 'fine_tune', 'evaluate'])

    args = parser.parse_args()
    if args.part == 'preprocess':
        print(bcolors.OKCYAN + "> This is the preprocess part.")
        test_preprocess()

    elif args.part == 'setup':
        print(bcolors.OKCYAN + "> This is the setup part.")
        test_setup()

    elif args.part == 'inference':
        print(bcolors.OKCYAN + "> This is the inference part.")
        test_inference()

    elif args.part == 'prepare_dataset':
        print(bcolors.OKCYAN + "> This is the prepare dataset part.")
        test_prepare_dataset()

    elif args.part == 'fine_tune':
        print(bcolors.OKCYAN + "> This is the fine-tune part.")
        test_fine_tune()

    elif args.part == 'evaluate':
        print(bcolors.OKCYAN + "> This is the evaluate part.")
        test_evaluate()
