'''
print real samples from the text with same number of characters, formating
as samples from the lstm model, to get sense of how the true text samples
should look'''

import os
import numpy as np
from pprint import pprint

import textwrap

import re

def main():

    directory = "books/"
    dir_list = os.listdir(directory)

    print("Here are the books available to read: ")
    pprint(dir_list)
    print()

    book = input("Book to read ? ")
    filename = directory + book + '.txt'
    print(f"Reading {filename}")
    print()

    if not os.path.isfile(filename):
        print(f"No text by the title: {book}")
        exit()

    with open(filename,'r') as file:
        data = file.read()

    # moderate preprocessing to reduce number of characters

    data = re.sub(r"[^a-zA-Z0-9.,?!:;“”'`’ ]+", " ", data)

    data = re.sub(r"[\s]+"," ",data)

    chars = set(data)

    char_to_ix = {ch:i for i,ch in enumerate(chars)}

    print(f'{filename} has {len(data)} characters')
    print()
    print(f'Will train on {len(chars)} unique characters')

    for i in range(8):
        print('-'*100)
        c_rand = np.random.randint(len(data)//3) + len(data)//3

        text_blob = data[c_rand:c_rand+300]

        wrap_list = textwrap.wrap(text_blob,90)

        wrap_list = [line.center(100) for line in wrap_list]

        print(f'Excerpt from: {book.replace("_"," ").upper()} '.center(100))
        print()
        print()
        print('\n'.join(wrap_list))
        print('-'*100)

if __name__ == '__main__':

    main()
