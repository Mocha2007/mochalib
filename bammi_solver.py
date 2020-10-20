from copy import deepcopy as copy
from typing import List

max_depth = 1

tiles = []

class Tile:
	def __init__(self, size: int, owner: int = -1):
		self.max_size = size
		self.current_size = 0
		self.adjacent = set() # type: Set[Tile]
		self.owner = owner # -1 = none, 0 = AI, 1 = player
		tiles.append(self)
	# speshul
	def __str__(self):
		return "<Tile " + str(tiles.index(self)) + ">"
	# non-static methods
	def click(self):
		if self.current_size == self.max_size:
			self.explode()
		else:
			self.current_size += 1
	def explode(self):
		self.current_size = 1
		for t in self.adjacent:
			t.owner = self.owner
			t.click()
	def link(self, other): # other is Tile
		self.adjacent.add(other)
		other.adjacent.add(self)

def get_board_for_move(board: List[Tile], move: Tile) -> List[Tile]:
	i = board.index(move)
	board = copy(board)
	board[i].click()
	return board

def valid_moves_for(board: List[Tile], player: int) -> List[Tile]:
	return [t for t in board if t.owner == player or t.owner == -1]

def rate_board_for_player(board: List[Tile], player: int) -> int:
	return len(valid_moves_for(board, player))

def best_player_move(board: List[Tile], player: int, depth: int = max_depth) -> Tile:
	# todo
	svm = valid_moves_for(board, player)
	svm.sort(key=lambda m: rate_board_for_player(
			get_board_for_move(board, m), player
	))
	return svm[0] if len(svm) else None

def rate_move_for_player(board: List[Tile], move: Tile, player: int, depth: int = max_depth) -> int:
	if depth == 0: # no opponent moves
		return rate_board_for_player(board, player)
	if depth == 1: # one opponent move
		other_player = 1-player
		other_player_moves = valid_moves_for(board, other_player)
		other_player_move_scores = [rate_move_for_player(board, m, player, depth-1) for m in other_player_moves]
		return min(other_player_move_scores)
	# else opponent + this + ...
	raise NotImplementedError

# tile list
tile_0_0 = Tile(2)
tile_0_2 = Tile(3)
tile_0_3 = Tile(3)
tile_0_4 = Tile(2)
tile_1_0 = Tile(4)
tile_1_2 = Tile(6)
tile_1_3 = Tile(2)
tile_2_0 = Tile(2)
tile_3_0 = Tile(3)
tile_3_1 = Tile(4)
tile_3_2 = Tile(4)
tile_3_3 = Tile(4)
tile_4_0 = Tile(4)
tile_4_3 = Tile(3)
tile_4_4 = Tile(2)

# connections
tile_0_0.link(tile_0_2)
tile_0_0.link(tile_1_0)

tile_0_2.link(tile_0_3)
tile_0_2.link(tile_1_2)

tile_0_3.link(tile_0_4)
tile_0_3.link(tile_1_3)

tile_1_0.link(tile_1_2)
tile_1_0.link(tile_2_0)
tile_1_0.link(tile_3_1)

tile_1_2.link(tile_1_3)
tile_1_2.link(tile_3_2)
tile_1_2.link(tile_3_3)

tile_2_0.link(tile_3_0)

tile_3_0.link(tile_3_1)
tile_3_0.link(tile_4_0)

tile_3_1.link(tile_3_2)
tile_3_1.link(tile_4_0)

tile_3_2.link(tile_3_3)
tile_3_2.link(tile_4_0)

tile_3_3.link(tile_4_3)
tile_3_3.link(tile_4_4)

tile_4_0.link(tile_4_3)

tile_4_3.link(tile_4_4)

# compute
print(best_player_move(tiles, 1))