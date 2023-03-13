#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import AutoModelForCausalLM, AutoTokenizer
import datetime
import torch
import re

import requests

'''

Contributed by SagsMug. Thank you SagsMug.
https://github.com/oobabooga/text-generation-webui/pull/175

'''

import asyncio
import json
import random
import string

import websockets


def random_hash():
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for i in range(9))

params = {}

async def run(context):
    server = "127.0.0.1"
    params = context
#        'max_new_tokens': 200,
#        'do_sample': True,
#        'temperature': 0.5,
#        'top_p': 0.9,
#        'typical_p': 1,
#        'repetition_penalty': 1.05,
#        'top_k': 0,
#        'min_length': 0,
#        'no_repeat_ngram_size': 0,
#        'num_beams': 1,
#        'penalty_alpha': 0,
#        'length_penalty': 1,
#        'early_stopping': False,
#    }
    session = random_hash()

    n = 0

    async with websockets.connect(f"ws://{server}:7860/queue/join") as websocket:
        while content := json.loads(await websocket.recv()):
            #Python3.10 syntax, replace with if elif on older
            match content["msg"]:
                case "send_hash":
                    await websocket.send(json.dumps({
                        "session_hash": session,
                        "fn_index": 7
                    }))
                case "estimation":
                    pass
                case "send_data":
                    await websocket.send(json.dumps({
                        "session_hash": session,
                        "fn_index": 7,
                        "data": [
                            params["prompt"],
                            params['max_new_tokens'],
                            params['do_sample'],
                            params['temperature'],
                            params['top_p'],
                            params['typical_p'],
                            params['repetition_penalty'],
                            params['top_k'],
                            params['min_length'],
                            params['no_repeat_ngram_size'],
                            params['num_beams'],
                            params['penalty_alpha'],
                            params['length_penalty'],
                            params['early_stopping'],
                        ]
                    }))
                case "process_starts":
                    pass
                case "process_generating" | "process_completed":
                    ret_me = content["output"]["data"][0]
                    do_it = False
                    print(ret_me[len(params["prompt"]):])
                    if "Human (" in ret_me[len(params["prompt"]):]:
                        ret_me = ret_me[:ret_me.rindex("Human (") - 1]
                        do_it = True
                    yield ret_me
#                    print(ret_me)
                    if do_it:
                        break
#                    print(content["output"]["data"][0])
                    # You can search for your desired end indicator and 
                    #  stop generation by closing the websocket here
                    if (content["msg"] == "process_completed"):
                        break

#prompt = "What I would like to say is the following: "

async def get_result_stream(params):
    s = "PLACEHOLDER"
    async for response in run(params):
        if s == response:
            break
        s = response
        yield response
        await asyncio.sleep(1)

async def get_result(params):
    s = "PLACEHOLDER"
    async for response in run(params):
        if s == response:
            break
#            pass
        s = response
        # Print intermediate steps
#        print(response)

    # Print final result
#    print(response)
    print("LOL420\n\n" + s + "\n\nLOL420\nA")
    return s

server = "127.0.0.1"

#model4 = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1", device_map="auto", load_in_8bit=True)
#tokenizer4 = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")

print("ready")

import threading

class RunThread(threading.Thread):
    def __init__(self, func, args, kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        super().__init__()

    def run(self):
        self.result = asyncio.run(self.func(*self.args, **self.kwargs))

def run_async(func, *args, **kwargs):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        thread = RunThread(func, args, kwargs)
        thread.start()
        thread.join()
        return thread.result
    else:
        return asyncio.run(func(*args, **kwargs))

def predict(input, temperature=0.7,top_p=0.01,top_k=40,max_tokens=500,no_repeat_ngram_size=0,num_beams=1,do_sample=True,length_penalty=5):
    s = input

    params = {
        'prompt': s,
        'max_new_tokens': max_tokens,
        'do_sample': do_sample,
        'temperature': temperature,
        'top_p': top_p,
        'typical_p': 1,
        'repetition_penalty': 1.1,
        'top_k': top_k,
        'min_length': 10,
        'no_repeat_ngram_size': no_repeat_ngram_size,
        'num_beams': num_beams,
        'penalty_alpha': 0,
        'length_penalty': length_penalty,
        'early_stopping': True,
    }

    return run_async(get_result, params)

async def predict_stream(input, temperature=0.7,top_p=0.01,top_k=40,max_tokens=500,no_repeat_ngram_size=0,num_beams=1,do_sample=True,length_penalty=5):
    s = input

    params = {
        'prompt': s,
        'max_new_tokens': max_tokens,
        'do_sample': do_sample,
        'temperature': temperature,
        'top_p': top_p,
        'typical_p': 1,
        'repetition_penalty': 1.1,
        'top_k': top_k,
        'min_length': 10,
        'no_repeat_ngram_size': no_repeat_ngram_size,
        'num_beams': num_beams,
        'penalty_alpha': 0,
        'length_penalty': length_penalty,
        'early_stopping': True,
    }

    async for i in get_result_stream(params):
        yield i

def predict2(input, temperature=0.7,top_p=1,top_k=0,max_tokens=64,no_repeat_ngram_size=1,num_beams=1,do_sample=True):
    s = input

    input_ids = tokenizer4.encode(str(s), return_tensors="pt").cuda()
    response = model4.generate(input_ids, min_length = 10,
                         max_new_tokens=int(max_tokens),
                         top_k=int(top_k),
                         top_p=float(top_p),
                         temperature=float(temperature),
                         no_repeat_ngram_size=int(no_repeat_ngram_size),
                         num_beams = int(num_beams),
                         do_sample = bool(do_sample),
                         )

    response2 = tokenizer4.decode(response[0])

    return response2

def predict3(input, temperature=0.7,top_p=0.01,top_k=40,max_tokens=64,no_repeat_ngram_size=0,num_beams=1,do_sample=True,length_penalty=5):
    s = input

    params = {
        'max_new_tokens': max_tokens,
        'do_sample': do_sample,
        'temperature': temperature,
        'top_p': top_p,
        'typical_p': 1,
        'repetition_penalty': 1.0,
        'top_k': top_k,
        'min_length': 10,
        'no_repeat_ngram_size': no_repeat_ngram_size,
        'num_beams': num_beams,
        'penalty_alpha': 0,
        'length_penalty': length_penalty,
        'early_stopping': True,
    }

#    print(params)

    response = requests.post(f"http://{server}:7860/run/textgen", json={
        "data": [
            s,
            params['max_new_tokens'],
            params['do_sample'],
            params['temperature'],
            params['top_p'],
            params['typical_p'],
            params['repetition_penalty'],
            params['top_k'],
            params['min_length'],
            params['no_repeat_ngram_size'],
            params['num_beams'],
            params['penalty_alpha'],
            params['length_penalty'],
            params['early_stopping'],
        ]
    }).json()
    
    reply = response["data"][0]

    return reply

prompt = open("prompt.txt", "r").read().format(date=str(datetime.datetime.now()))
input_text = open("input.txt", "r").read().split("\n")[0].format(date=str(datetime.datetime.now()))

s = prompt

threads = {}

def reset(thread_id):
    global threads
    prompt = open("prompt.txt", "r").read().format(date=str(datetime.datetime.now()))
    threads[thread_id] = prompt

def full_history(thread_id):
    return threads[thread_id]

def load_history(thread_id, history):
    threads[thread_id] = history

def send_message(txt, thread_id):
    global threads
    input_text = open("input.txt", "r").read().split("\n")[0].format(date=str(datetime.datetime.now()))
    if not thread_id in threads:
#        threads[thread_id] = ""
        reset(thread_id)
    assistant_input_text = open("assistant_input.txt", "r").read().split("\n")[0].format(date=str(datetime.datetime.now()))
    threads[thread_id] += input_text + txt + "\n" + assistant_input_text
    tmp = predict(threads[thread_id], max_tokens=500, temperature=0.7, top_p=0.01, top_k=40, )
    tmp = (tmp[len(threads[thread_id]):])
    print(tmp)
#    tmp = tmp.split("\n")[0]
    threads[thread_id] += tmp + "\n"
    return tmp
#    print("Assistant:" + tmp)

async def send_message_stream(txt, thread_id):
    global threads
    input_text = open("input.txt", "r").read().split("\n")[0].format(date=str(datetime.datetime.now()))
    if not thread_id in threads:
#        threads[thread_id] = ""
        reset(thread_id)
    assistant_input_text = open("assistant_input.txt", "r").read().split("\n")[0].format(date=str(datetime.datetime.now()))
    threads[thread_id] += input_text + txt + "\n" + assistant_input_text
    tmp = ""
    async for i in predict_stream(threads[thread_id], max_tokens=500, temperature=0.7, top_p=0.01, top_k=40, ):
        yield i[len(threads[thread_id]):]
        tmp = i[len(threads[thread_id]):]
#    tmp = (tmp[len(threads[thread_id]):])
#    print(tmp)
#    tmp = tmp.split("\n")[0]
    threads[thread_id] += tmp + "\n"
#    return tmp
#    print("Assistant:" + tmp)
