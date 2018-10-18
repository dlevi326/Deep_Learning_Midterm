import json, random, re, os

# Gets text from json file
def getText(filename):
    with open(filename, 'rb') as f:
        file = json.load(f)
    text = file['text']
    text = text.replace('\n','')  # removing newlines '\n' from text
    textlist = list(map(str.strip, re.split(r"[.!?](?!$)", text)))  # split text into sentences
    if len(textlist) > 1:  # if there are more than one sentence in text
        return text, random.choice(textlist)

# Loops through folder and calls getText for each json file
def loopFiles(directory):

    for file in os.listdir(directory):
        filename = directory + file
        filename = str(filename)

        if filename.endswith(".json"):
            print(getText(filename))


data_folder = (os.getcwd() + "\\newsfiles\\")

loopFiles(data_folder)
