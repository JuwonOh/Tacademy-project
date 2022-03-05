from fastapi import APIRouter
from rpc.rpc_client import NLPpredicting
from schema import Fib
from rpc.rpc_client import FibonacciRpcClient

router = APIRouter(prefix='/api_with_rabbitmq')

@router.get('/')
def test():
    return 'API is running'

@router.post('/calculate')
async def calculate_fibonacci(inputData:Fib):
    fibonacci_rpc = FibonacciRpcClient()

    print(" [x] Requesting fib(%s)" % inputData.fibNumber)
    response = fibonacci_rpc.call(inputData.fibNumber)
    print(" [.] Got %r" % response)

    return response

@router.post('/predict_nlp')
async def predict_NLP(n: str, m: str):
    nlp_predict = NLPpredicting()

    print("[x] Requesting nlp")
    response = nlp_predict.call(n)
    response1 = nlp_predict.call(m)
    print(" [.] Got first reponse : %d" % response)
    print(" [.] Got second reponse : %d" % response1)


    return response