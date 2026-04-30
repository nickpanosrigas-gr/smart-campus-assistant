from typing import Literal

# 1. Room Literal
CampusRooms = Literal[
    'parkin.c', 'parkin.b', 'data_center', 'entrance', 'restaurant', 
    '1.1', '1.2', 'kitchen', '2.1', '2.2', '2.3', '2.4', 
    '3.7', '3.8', '3.9', '4.9', '5.6', '5.7', 'roof'
]

# 2. Timeframe Literal
Timeframes = Literal[
    'now', '2h', '24h', '7d', '30d', '90d'
]