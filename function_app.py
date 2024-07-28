import os
from transformers import GPT2LMHeadModel
import torch
from tokenizer import CharTokenizer
import azure.functions as func
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

MODEL_PATH = os.environ.get('MODEL_PATH', './model/checkpoint-384000')
VOCAB_FILE = os.environ.get('VOCAB_FILE', './tokenizer/vocab.json')

tokenizer = None
model = None

def initialize():
    global tokenizer, model
    try:
        tokenizer = CharTokenizer(
            vocab_file=VOCAB_FILE, 
            bos_token="<BOS>",
            eos_token="<EOS>",
            sep_token="<SEP>",
            unk_token="<UNK>",
            pad_token="<PAD>"
        )
        tokenizer.padding_side = "left"

        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        model.eval()
        logger.info("Model and tokenizer initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing model or tokenizer: {str(e)}")
        raise

def compute_log_likelihood(pw):
    try:
        forgen_result = tokenizer.encode_forgen(pw)
        input_id = forgen_result.view([1, -1])
        with torch.no_grad():
            outputs = model(input_id, labels=input_id)
            log_likelihood = outputs.loss.item()
        return log_likelihood
    except Exception as e:
        logger.error(f"Error computing log likelihood: {str(e)}")
        raise

@app.route(route="http_trigger")
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