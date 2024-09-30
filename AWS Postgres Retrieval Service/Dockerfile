FROM public.ecr.aws/lambda/python:3.12

COPY .env ${LAMBDA_TASK_ROOT}
COPY requirements.txt ${LAMBDA_TASK_ROOT}
COPY ../all-MiniLM-l6-v2/ ${LAMBDA_TASK_ROOT}/all-MiniLM-l6-v2/
COPY source/* ${LAMBDA_TASK_ROOT}

RUN pip install -r requirements.txt

CMD [ "lambda_function.lambda_handler" ]
