import json


# Parses facebook message data
# Encoding with emoji support

def parse_obj(dct):
    print(dct)
    for key in dct:
        try:
            dct[key] = dct[key].encode('latin_1').decode('utf-8')
        except AttributeError:
            dct[key] = dct[key]
        pass
    return dct


with open('messages.json', 'r', encoding='utf-8') as f:
    obj = ''.join(f.readlines())
    data = json.loads(obj, object_hook=parse_obj)
    messages_json = data['messages']

    # List comprehension to get all message content but return "" if there is no content
    messages_json = filter(lambda message: 'content' in message, messages_json)
    messages = [f"{message['sender_name']}: {message['content']}\n" for message in messages_json]

    # Write to file
    with open('messages.txt', 'w', encoding='utf-8') as f:
        f.writelines(messages)
