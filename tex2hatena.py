import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("path", help="text file including latex equations.")

args = parser.parse_args()
path = args.path

def convert(text):
    link_indices = [m.span() for m in re.finditer("\[[あ-んぁ-んァ-ン一-龥a-zA-Z0-9!-/:-@[-`{-~]*\]([a-zA-Z0-9!-/:-@[-`{-~]*)", text)][::-1]
    links = []
    for i,idx in enumerate(link_indices):   # 逆順になっていることに注意
        links.append(text[idx[0]:idx[1]])
        text = text[:idx[0]] + "@@@" + text[idx[1]:]

    text = text.replace("\\\\", "\\\\\\")
    text = text.replace("_", "\\_")
    text = text.replace("^", "\\^")
    text = text.replace("[", "\\[")
    text = text.replace("]", "\\]")
    text = text.replace("\\{", "\\\\{")
    text = text.replace("\\}", "\\\\}")
    text = text.replace("*", "\\*")

    # equations
    text = text.replace(r"$$", r"")
    N = len([m.span() for m in re.finditer("$$", text)][::-1])
    if N > 0:
        text = """[tex: \displaystyle]
""" + text
    
    indices = [m.span() for m in re.finditer("\$", text)][::-1]
    for i,idx in enumerate(indices):   # 逆順になっていることに注意
        if i % 2 == 0:
            # $ -> ]
            text = text[:idx[0]] + "]" + text[idx[1]:]
        elif i % 2 == 1:
            text = text[:idx[0]] + "[tex: \\displaystyle " + text[idx[1]:]


    link_indices = [m.span() for m in re.finditer("@@@", text)][::-1]
    for i,idx in enumerate(link_indices):   # 逆順になっていることに注意
        text = text[:idx[0]] + links[i] + text[idx[1]:]

    return text


with open(path) as f:
    text = f.read()

newtext = convert(text)

with open("hatena.txt", "w") as f:
    f.write(newtext)


