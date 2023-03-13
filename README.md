# LLaMA_chatbot

## ***WARNING!!!***
THIS CODE IS FUCKING AWFUL. DO NOT EXPECT IT TO WORK OUT OF THE BOX. I'M GOING
TO ATTEMPT TO IMPROVE IT OVER THE NEXT WHILE, BUT FOR NOW, IT'S SHIT.

IF YOU CAN GET IT TO WORK, YOU DESERVE A COOKIE.
CHEERS.

## "general install guide"
* install text-generation-webui from oogabooga
* download the LLaMA weights, whatever is the largest that'll fit on your GPU(s)
* run text-generation-webui for that model, `--listen`
* make a venv `python -m venv venv`
* install requirements `venv/bin/pip install -r requirements.txt`
* set env `MATRIX_{SERVER,USER,PASSWORD,ROOM}` to applicable vals, ex:
  `https://synapse.example.com`, `@AzureDiamond:example.com`, `hunter2`, 
  fancy room ID that's like `\![A-Za-z]*:example\.com`) respectively
* run client.py `venv/bin/python client.py`
* enjoy
