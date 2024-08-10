import json
import os
from transformers import GPT2LMHeadModel
import torch
from tokenizer import CharTokenizer
import azure.functions as func
import logging
import random
import re
import string
from itertools import permutations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

MODEL_PATH = os.environ.get('MODEL_PATH', './model/')
VOCAB_FILE = os.environ.get('VOCAB_FILE', './tokenizer/vocab.json')
MAX_LEN = 32

tokenizer = None
model = None

def initialize():
    global tokenizer, model
    try:
        logger.info("Initializing tokenizer...")
        tokenizer = CharTokenizer(
            vocab_file=VOCAB_FILE, 
            bos_token="<BOS>",
            eos_token="<EOS>",
            sep_token="<SEP>",
            unk_token="<UNK>",
            pad_token="<PAD>"
        )
        tokenizer.padding_side = "left"
        logger.info("Tokenizer initialized successfully.")

        logger.info("Loading model from path: %s", MODEL_PATH)
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        model.resize_token_embeddings(len(tokenizer))
        model.eval()
        logger.info("Model loaded and set to evaluation mode.")
    except Exception as e:
        logger.error(f"Error initializing model or tokenizer: {str(e)}")
        raise

def get_pattern(password: str):
    result = []
    current_type = None
    current_length = 0

    for char in password:
        if char.isalpha():
            if current_type == 'L':
                current_length += 1
            else:
                if current_type:
                    result.append(current_type + str(current_length))
                current_type = 'L'
                current_length = 1
        elif char.isdigit():
            if current_type == 'N':
                current_length += 1
            else:
                if current_type:
                    result.append(current_type + str(current_length))
                current_type = 'N'
                current_length = 1
        else:
            if current_type == 'S':
                current_length += 1
            else:
                if current_type:
                    result.append(current_type + str(current_length))
                current_type = 'S'
                current_length = 1

    if current_type:
        result.append(current_type + str(current_length))
    return result

def generate_password_pattern(permutation, pw_len):
    parts = permutation.split()
    counts = {part[0]: int(part[1:]) for part in parts}
    max_len = max(8, pw_len)
    
    for letter in 'LNS':
        if letter not in counts:
            counts[letter] = 0
    
    total = sum(counts.values())
    while sum(counts.values()) != max_len or min(counts.values()) == 0:
        if sum(counts.values()) < max_len:
            # Add to missing or smallest count
            candidates = [l for l in 'LNS' if counts[l] == min(counts.values())]
            letter = random.choice(candidates)
            counts[letter] += 1
        else:
            # Subtract from largest count
            candidates = [l for l in 'LNS' if counts[l] == max(counts.values()) and counts[l] > 1]
            if candidates:
                letter = random.choice(candidates)
                counts[letter] -= 1
            else:
                # If we can't subtract, redistribute
                for l in 'LNS':
                    if counts[l] > 1:
                        counts[l] -= 1
                        break
    
    parts = [f"{letter}{count}" for letter, count in counts.items() if count > 0]
    all_permutations = list(permutations(parts))
    
    permutation_strings = [' '.join(perm) for perm in all_permutations]
    return permutation_strings

def ensure_case_diversity(password):
    if has_both_cases(password):
        return password
    
    password_chars = list(password)
    
    if not has_uppercase(password):
        add_uppercase(password_chars)
    
    if not has_lowercase(password):
        add_lowercase(password_chars)
    
    return ''.join(password_chars)

def has_both_cases(string):
    return has_uppercase(string) and has_lowercase(string)

def has_uppercase(string):
    return any(char.isupper() for char in string)

def has_lowercase(string):
    return any(char.islower() for char in string)

def add_uppercase(chars):
    change_random_char(chars, str.islower, str.upper)

def add_lowercase(chars):
    change_random_char(chars, str.isupper, str.lower)

def change_random_char(chars, condition, transform):
    allowed_chars = string.ascii_letters
    eligible_indices = [i for i, char in enumerate(chars) if condition(char)]
    if eligible_indices:
        index = random.choice(eligible_indices)
        transformed_char = transform(chars[index])
        # Ensure the transformed character is in the allowed set
        while transformed_char not in allowed_chars:
            transformed_char = transform(random.choice(list(allowed_chars)))
        chars[index] = transformed_char

def valid_format(password):
    pattern = r'^(?=.*[!@#$%^&*()_+\-=[\]{}|;:\'",.<>/?`~])(?=.*\d).+$'
    return bool(re.match(pattern, password)) and len(password) >= 8

def compute_log_likelihood(pw):
    try:
        logger.info("Received password: %s", pw)
        input_pw = ' '.join(get_pattern(pw)) + ' <SEP> ' + ' '.join(list(pw))
        logger.info("Encoding: %s", input_pw)
        forgen_result = tokenizer.encode_forgen(input_pw)
        input_id = forgen_result.view([1, -1])

        with torch.no_grad():
            outputs = model(input_id, labels=input_id)
            log_likelihood = outputs.loss.item()

        return log_likelihood
    except Exception as e:
        logger.error(f"Error computing log likelihood: {str(e)}")
        raise

def generate_variants(pw):
    passwords = set()
    ip = ' '.join(get_pattern(pw))
    fps = generate_password_pattern(ip, len(pw))

    inputs = set()
    logger.info(f"{pw} patterns: {fps}")
    for fp in fps:
        inputs.add(fp + ' <SEP> ' + ' '.join(list(pw)))

    tokenizer_forgen_results = [tokenizer.encode_forgen(input_text) for input_text in inputs]    
    for tokenizer_forgen_result in tokenizer_forgen_results:
        input_ids=tokenizer_forgen_result.view([1, -1])
        outputs = model.generate(
            input_ids=input_ids,
            pad_token_id=tokenizer.pad_token_id,
            max_length=MAX_LEN,
            do_sample=True,
            num_return_sequences=10
        )
        decoded_outputs = tokenizer.batch_decode(outputs)
        for output in decoded_outputs:
            pattern, pw_variant = output.split(' ', 1)
            if valid_format(pw_variant):
                passwords.add(ensure_case_diversity(pw_variant))
    
    return passwords

@app.function_name(name="HttpTrigger1")
@app.route(route="hello")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logger.info('Python HTTP trigger function processed a request.')

    if tokenizer is None or model is None:
        initialize()

    password = req.params.get('password')
    if not password:
        try:
            req_body = req.get_json()
            password = req_body.get('password')
        except ValueError:
            pass

    if not password:
        return func.HttpResponse(
            json.dumps({"error": "Please pass a password in the query string or in the request body."}),
            mimetype="application/json",
            status_code=400
        )

    try:
        model.resize_token_embeddings(len(tokenizer))
        log_likelihood = compute_log_likelihood(password)
        variants = generate_variants(password)
        return func.HttpResponse(
            json.dumps({"log_likelihood": f"{log_likelihood:.6f}", "variants": list(variants)}),
            mimetype="application/json",
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "An error occurred while processing your request."}),
            mimetype="application/json",
            status_code=500
        )

initialize()
