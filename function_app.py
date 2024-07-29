import os
from transformers import GPT2LMHeadModel
import torch
from tokenizer import CharTokenizer
import azure.functions as func
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

MODEL_PATH = os.environ.get('MODEL_PATH', './model/')
VOCAB_FILE = os.environ.get('VOCAB_FILE', './tokenizer/vocab.json')

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
            "Please pass a password in the query string or in the request body.",
            status_code=400
        )

    try:
        log_likelihood = compute_log_likelihood(password)
        return func.HttpResponse(
            f"Password Log Likelihood: {log_likelihood:.6f}",
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return func.HttpResponse(
            "An error occurred while processing your request.",
            status_code=500
        )

initialize()
