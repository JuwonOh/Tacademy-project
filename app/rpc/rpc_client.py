import pika
import uuid

class FibonacciRpcClient:
# class NLPpredicting:

    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost')
        )
        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True
        )

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, n):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='rpc_queue',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=str(n)
        )
        while self.response is None:
            self.connection.process_data_events()
        
        return int(self.response)



class NLPpredicting:
"""
rpc로 원격 인스턴스에 작업을 할당하는 코드와 원격 인스턴스에서 처리한 결과를 리턴받는 클래스
"""
    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost')
        )
        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue=result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True
        )
    
    def on_response(self, ch, method, props, body):
        """
        요청에 대해 받아야할 응답
        """
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, n: str):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        print(self.corr_id)
        self.channel.basic_publish(
            exchange='',
            routing_key='rpc_queue',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body = n # str(n)
        )
        while self.response is None:
            self.connection.process_data_events()

        return int(self.response)