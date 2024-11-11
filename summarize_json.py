#!/usr/bin/env python

#https://gist.github.com/mbafford/5a0ff6675b92ea1a6d80

# Gives a quick summary of the structure of a JSON file, including the keys, object types, and
# leaf node types. Provides a count of each data type so you can quickly tell which data points
# are common.

from collections import defaultdict


def yieldkeys( data, parent_key = None ):
    parent_key = '%s.' % ( parent_key ) if parent_key else ''

    if isinstance( data, list ):
        for i,item in enumerate( data ):
            for y in yieldkeys( item, '%s[]' % ( parent_key ) ):
                yield(y)
    elif isinstance( data, dict ):
        for i, item in data.items():
            for y in yieldkeys( item, '%s{%s}' % ( parent_key, i ) ):
                yield(y)
    else:
        yield( '%s%s' % ( parent_key, type(data).__name__ ) )


def summarize_json(data):
    keycount = defaultdict(lambda: 0)
    for a in yieldkeys(data):
        keycount[a] += 1

    for key in sorted( keycount.keys() ):
        print("%4d %s" % ( keycount[key], key ))
