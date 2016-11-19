#!/usr/bin/python

import sys

for line in sys.stdin:
  line = line.strip()
  ids  = line.split()

  user    = ids[0]
  friends = ids[1:]

  # Sort the nodes to keep edge (i.e. key) unique
  edge = (user, friends[0]) if user < friends[0] else (friends[0], user)

  # For each edge between user and some friend, print all other friends
  for friend in friends:
    edge = (user, friend) if user < friend else (friend, user)     
    for other in friends:
      if friend != other:
        print('{0} {1} {2} {3}'.format(edge[0], edge[1], user, other))
