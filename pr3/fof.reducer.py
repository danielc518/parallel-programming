#!/usr/bin/python

import sys

map = dict()

# Shuffle
for line in sys.stdin:
  line = line.strip()
  ids  = line.split()

  edge = (ids[0], ids[1]) # Edge of friends
  
  if edge not in map:
    map[edge] = dict()

  user = ids[2]
  friend = ids[3]

  if user not in map[edge]:
    map[edge][user] = set() # Other friends of 'user'

  map[edge][user].add(friend)

# Reduce
for edge, friends in map.items():
  userA = edge[0]
  userB = edge[1]

  # Get other friends of userA and userB
  friendsOfA = friends[userA] if userA in friends else set()
  friendsOfB = friends[userB] if userB in friends else set()
  
  # Find common friends of userA and userB
  commonFriends = set.intersection(friendsOfA, friendsOfB)
  for commonFriend in commonFriends:
    print('{0} {1} {2}'.format(commonFriend, userA, userB))
      

