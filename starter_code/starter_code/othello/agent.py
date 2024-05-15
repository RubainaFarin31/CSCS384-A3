"""
An AI player for Othello. 
"""

## Assisted with debugging using ChatGPT

import random
import sys
import time

# You can use the functions in othello_shared to write your AI
from othello_shared import find_lines, get_possible_moves, get_score, play_move

def eprint(*args, **kwargs): #you can use this for debugging, as it will print to sterr and not stdout
    print(*args, file=sys.stderr, **kwargs)
    
# Method to compute utility value of terminal state
def compute_utility(board, color):
    scores = get_score(board)
    return scores[0] - scores[1] if color == 1 else scores[1] - scores[0]


# Better heuristic value of board
def compute_heuristic(board, color):
    # Count the number of coins for the given color and opponent's color
    color_coins = sum(row.count(color) for row in board)
    opp_coins = sum(row.count(2 if color == 1 else 1) for row in board)

    # Count the number of possible moves for the given color and opponent's color
    color_moves = len(get_possible_moves(board, color))
    opp_moves = len(get_possible_moves(board, 2 if color == 1 else 1))

    # Compute the parity, mobility, and edge components of the heuristic
    parity = 100 * (color_coins - opp_coins) / max(color_coins + opp_coins, 1)
    mobility = 100 * (color_moves - opp_moves) / max(color_moves + opp_moves, 1)

    # Count coins on the edges of the board
    edge = sum(1 for r in (board[0], board[-1]) for c in r if c == color)

    # Combine the components using weights and return the heuristic value
    return compute_utility(board, color) + parity + mobility + edge


############ MINIMAX ###############################

normal_cache = {}

#Helper functions

def get_opponent_color(current_color):
    return 2 if current_color == 1 else 1

def is_terminal_state(moves, remaining_limit):
    return not moves or remaining_limit == 0

#####

def minimax_min_node(board, color, limit, caching=0):
    # Helper function to update the cache if caching is enabled
    def update_cache(key, result):
        if caching:
            normal_cache[key] = result

    # Convert the board and color into a string key for caching
    key = str((board, color))
    # Check if caching is enabled and the key exists in the cache
    if caching and key in normal_cache:
        # Return the cached result if available
        return normal_cache[key]

    # Determine the opponent's color
    opponent_color = get_opponent_color(color)
    # Get possible moves for the opponent
    possible_moves = get_possible_moves(board, opponent_color)

    # Check if we have reached a terminal state
    if is_terminal_state(possible_moves, limit):
        # Return utility for the current state if it's terminal
        return (None, compute_utility(board, color))

    # Initialize variables to track best move and minimum utility
    best_move = None
    min_utility = float("inf")

    # Iterate through possible moves to find the one with minimum utility
    for move in possible_moves:
        # Simulate opponent's move
        new_board = play_move(board, opponent_color, move[0], move[1])
        # Get utility for the resulting board state
        _, utility = minimax_max_node(new_board, color, limit - 1, caching)
        # Update best move and minimum utility if a better move is found
        if utility < min_utility:
            best_move = move
            min_utility = utility
            # If the minimum utility is reached (worst case), no need to continue
            if min_utility == float("-inf"):
                break

    # Store the best move and its utility in the cache
    result = (best_move, min_utility)
    update_cache(key, result)

    return result


def minimax_max_node(board, color, limit, caching=0):
    # Helper function to update the cache if caching is enabled
    def update_cache(key, result):
        if caching:
            normal_cache[key] = result

    # Convert the board and color into a string key for caching
    key = str((board, color))
    # Check if caching is enabled and the key exists in the cache
    if caching and key in normal_cache:
        # Return the cached result if available
        return normal_cache[key]

    # Initialize best move and maximum utility
    best_move = None
    max_utility = float("-inf")

    # Check if the game has ended or the limit has been reached
    moves = get_possible_moves(board, color)
    if not moves or limit == 0:
        return (None, compute_utility(board, color))

    # Perform a depth-limited search for the best move
    for move in moves:
        new_board = play_move(board, color, move[0], move[1])
        _, utility = minimax_min_node(new_board, color, limit - 1, caching)

        # Update best move and maximum utility if a better move is found
        if utility > max_utility:
            best_move = move
            max_utility = utility

            # Break early if maximum possible utility is reached
            if max_utility == float("inf"):
                break

    # Cache the result and return
    result = (best_move, max_utility)
    update_cache(key, result)
    return result


def select_move_minimax(board, color, limit, caching = 0):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    """
    #IMPLEMENT (and replace the line below)
    result = minimax_max_node(board, color, limit, caching)
    return result[0]


############ ALPHA-BETA PRUNING #####################
def alphabeta_min_node(board, color, alpha, beta, limit, cache=0, order=0):
    # Determine opponent's color
    opp_color = 1 if color == 2 else 2
    key = (board, color, alpha, beta)
    # Check if caching is enabled and the current state is in memory
    if cache and key in mem:
        return mem[key]

    # Get possible moves for the opponent
    moves = get_possible_moves(board, opp_color)
    # Base case: no moves left or depth limit reached
    if not moves or limit == 0:
        return (None, compute_utility(board, color))
    
    best_move, min_util = None, float("inf")
    # Iterate through possible moves
    for move in moves:
        new_board = play_move(board, opp_color, *move)
        # Recursive call to maximize player's utility
        util = alphabeta_max_node(new_board, color, alpha, beta, limit-1, cache, order)[1]

        # Prune if the utility is less than or equal to alpha
        if util <= alpha:
            return (move, util)
        
        # Update beta value
        if util < beta:
            beta = util
        
        # Update best move and minimum utility
        if util < min_util:
            best_move, min_util = move, util

    # Cache the result if caching is enabled
    if cache:
        mem[key] = (best_move, min_util)
    return (best_move, min_util)

def alphabeta_max_node(board, color, alpha, beta, limit, cache=0, order=0):
    key = (board, color, alpha, beta)
    # Check if caching is enabled and the current state is in memory
    if cache and key in mem:
        return mem[key]

    # Get possible moves for the maximizing player
    moves = get_possible_moves(board, color)
    # Base case: no moves left or depth limit reached
    if not moves or limit == 0:
        return (None, compute_utility(board, color))
    
    best_move, max_util = None, float("-inf")
    # Iterate through possible moves
    for move in moves:
        new_board = play_move(board, color, *move)
        # Recursive call to minimize opponent's utility
        util = alphabeta_min_node(new_board, color, alpha, beta, limit-1, cache, order)[1]

        # Prune if the utility is greater than or equal to beta
        if util >= beta:
            return (move, util)
        
        # Update alpha value
        if util > alpha:
            alpha = util
        
        # Update best move and maximum utility
        if util > max_util:
            best_move, max_util = move, util

    # Cache the result if caching is enabled
    if cache:
        mem[key] = (best_move, max_util)
    return (best_move, max_util)

def select_move_alphabeta(board, color, limit, cache=0, order=0):
    # Entry point for selecting the best move using Alpha-Beta pruning
    return alphabeta_max_node(board, color, float("-inf"), float("inf"), limit, cache, order)[0]

# Initialize memory for caching results
mem = {}


####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Othello AI") # First line is the name of this AI
    arguments = input().split(",")
    
    color = int(arguments[0]) #Player color: 1 for dark (goes first), 2 for light. 
    limit = int(arguments[1]) #Depth limit
    minimax = int(arguments[2]) #Minimax or alpha beta
    caching = int(arguments[3]) #Caching 
    ordering = int(arguments[4]) #Node-ordering (for alpha-beta only)

    if (minimax == 1): eprint("Running MINIMAX")
    else: eprint("Running ALPHA-BETA")

    if (caching == 1): eprint("State Caching is ON")
    else: eprint("State Caching is OFF")

    if (ordering == 1): eprint("Node Ordering is ON")
    else: eprint("Node Ordering is OFF")

    if (limit == -1): eprint("Depth Limit is OFF")
    else: eprint("Depth Limit is ", limit)

    if (minimax == 1 and ordering == 1): eprint("Node Ordering should have no impact on Minimax")

    while True: # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL": # Game is over.
            print
        else:
            board = eval(input()) # Read in the input and turn it into a Python
                                  # object. The format is a list of rows. The
                                  # squares in each row are represented by
                                  # 0 : empty square
                                  # 1 : dark disk (player 1)
                                  # 2 : light disk (player 2)

            # Select the move and send it to the manager
            if (minimax == 1): #run this if the minimax flag is given
                movei, movej = select_move_minimax(board, color, limit, caching)
            else: #else run alphabeta
                movei, movej = select_move_alphabeta(board, color, limit, caching, ordering)
            
            print("{} {}".format(movei, movej))

if __name__ == "__main__":
    run_ai()
