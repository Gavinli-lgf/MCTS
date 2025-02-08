#!/usr/bin/env python
import random
import math
import hashlib
import logging
import argparse


"""
A quick Monte Carlo Tree Search implementation.  For more details on MCTS see See http://pubs.doc.ic.ac.uk/survey-mcts-methods/survey-mcts-methods.pdf

The State is a game where you have NUM_TURNS and at turn i you can make
a choice from an integeter [-2,2,3,-3]*(NUM_TURNS+1-i).  So for example in a game of 4 turns, on turn for turn 1 you can can choose from [-8,8,12,-12], and on turn 2 you can choose from [-6,6,9,-9].  At each turn the choosen number is accumulated into a aggregation value.  The goal of the game is for the accumulated value to be as close to 0 as possible.

The game is not very interesting but it allows one to study MCTS which is.  Some features 
of the example by design are that moves do not commute and early mistakes are more costly.  

In particular there are two models of best child that one can use 
论文参照:http://www.incompleteideas.net/609%20dropbox/other%20readings%20and%20resources/MCTS-survey.pdf
"""

#MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration. 
SCALAR=1/(2*math.sqrt(2.0))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('MyLogger')

# State定义了游戏的一个状态，包括当前值、剩余回合数和到达该状态的移动
class State():
	NUM_TURNS = 10		# 游戏的总回合数
	GOAL = 0			# 游戏目标值
	MOVES=[2,-2,3,-3]	# 每个回合可以选择的移动值
	MAX_VALUE= (5.0*(NUM_TURNS-1)*NUM_TURNS)/2	# 最大可能的值，用于归一化奖励
	num_moves=len(MOVES)						# 可选择的移动值的个数(也是一个节点最多可拓展的子节点数)

	# 输入:value 当前的状态值(游戏目标是让其接近0), moves 历史移动值序列, turn 总回合数;
	def __init__(self, value=0, moves=[], turn=NUM_TURNS):
		self.value=value
		self.turn=turn
		self.moves=moves

	# 每个回合都从"self.turn * self.MOVES"中随机选择一个移动值nextmove, 并以nextmove生成下一个State为next.
	def next_state(self):
		nextmove=random.choice([x*self.turn for x  in self.MOVES])
		next=State(self.value+nextmove, self.moves+[nextmove],self.turn-1)
		return next

	# 检查当前状态是否为终止状态（即剩余总回合为0）
	def terminal(self):
		if self.turn == 0:
			return True
		return False

	# 计算当前状态的奖励(用当前值与目标值的差异归一化到[0, 1]范围,即越self.value越接近self.GOAL, r越大)
	def reward(self):
		r = 1.0-(abs(self.value-self.GOAL)/self.MAX_VALUE)
		return r
	
	# 生成当前状态的哈希值，用于在集合或字典中存储状态
	def __hash__(self):
		return int(hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(),16)
	
	# 比较两个状态是否相等，基于它们的哈希值
	def __eq__(self,other):
		if hash(self)==hash(other):
			return True
		return False
	
	# 返回当前状态的字符串表示，便于调试和打印
	def __repr__(self):
		s="Value: %d; Moves: %s"%(self.value,self.moves)
		return s
	
# Node定义了一个表示蒙特卡洛树搜索（MCTS）树节点的类
class Node():
	# 输入:state 节点对应的游戏状态, parent 节点的父节点;
	def __init__(self, state, parent=None):
		self.visits=1		# 节点被访问的次数
		self.reward=0.0		# 节点的累积奖励
		self.state=state	# 见输入
		self.children=[]	# 节点的子节点列表
		self.parent=parent	# 见输入

	# 将child_state创建为当前节点的子节点Node，并将其添加到self.children列表中
	def add_child(self,child_state):
		child=Node(child_state,self)
		self.children.append(child)

	# 更新节点的累积奖励和访问次数
	def update(self,reward):
		self.reward+=reward
		self.visits+=1

	# 检查当前节点的子节点数是否等于最大子节点数，以确定当前节点是否被完全扩展。
	def fully_expanded(self, num_moves_lambda):
		num_moves = self.state.num_moves
		if num_moves_lambda != None:
		  num_moves = num_moves_lambda(self)
		if len(self.children)==num_moves:
			return True
		return False

	# 返回节点的字符串表示，便于调试和打印(子节点个数,访问次数,转态价值)
	def __repr__(self):
		s="Node; children: %d; visits: %d; reward: %f"%(len(self.children),self.visits,self.reward)
		return s

# 功能:基于"选择,拓展,仿真,回溯"实现的UCT的MCTS，用于在给定budget内找到最佳的子节点
# 输入:budget , root 搜索树的根节点, num_moves_lambda 可选函数用于动态确定节点的最大子节点数;
# 输出:返回经过budget次搜索后,root的BESTCHILD(即返回root的最优子节点,而不是最优叶子节点)
def UCTSEARCH(budget,root,num_moves_lambda = None):
	# "Selection,Expansion,Simulation,Backpropagation"循环迭代,共执行budget次(每10000次打印日志)。返回最终root的BESTCHILD.
	for iter in range(int(budget)):
		if iter%10000==9999:
			logger.info("simulation: %d"%iter)
			logger.info(root)
		front=TREEPOLICY(root, num_moves_lambda) # Selection,Expansion
		reward=DEFAULTPOLICY(front.state)		 # Simulation
		BACKUP(front,reward)					 # Backpropagation
	return BESTCHILD(root,0) # SCALAR设置为0,意味着选择root的子节点时只考虑exploit不考虑explore.

# 功能: Selection,Expansion,用于MCTS搜索树中选择和扩展节点,并返回最终选择节点; 输入:node 当前节点, num_moves_lambda 动态确定节点的最大子节点数;
def TREEPOLICY(node, num_moves_lambda):
	#a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
	# node没有子节点时，就拓展子节点并返回；node有子节点时若子节点未fully_expanded，则有50%的概率拓展并返回，也有50%概率从已有子节点中选择BESTCHILD.
	# 选择的终止条件是2个：选择到了terminal转态的节点；拓展了新的子节点。
	while node.state.terminal()==False:
		if len(node.children)==0:
			return EXPAND(node)
		elif random.uniform(0,1)<.5:
			node=BESTCHILD(node,SCALAR)
		else:
			if node.fully_expanded(num_moves_lambda)==False:	
				return EXPAND(node)
			else:
				node=BESTCHILD(node,SCALAR)
	return node

# 扩展当前节点,生成一个新的子节点,添加到当前节点的子节点列表中,并返回.(新拓展的子节点如果不是terminal状态,就不能与其他已存在的子节点重复)
def EXPAND(node):
	tried_children=[c.state for c in node.children]
	new_state=node.state.next_state()
	while new_state in tried_children and new_state.terminal()==False:
	    new_state=node.state.next_state()
	node.add_child(new_state)
	return node.children[-1]

#current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
# 功能:使用UCB1从当前节的子节点中选择一个最佳的子节点(注:是子节点不是叶子节点). 输入:node 当前节点, scalar UCB1中调整探索和利用之间平衡的参数.
def BESTCHILD(node,scalar):
	bestscore=0.0
	bestchildren=[]
	for c in node.children:
		exploit=c.reward/c.visits
		explore=math.sqrt(2.0*math.log(node.visits)/float(c.visits)) # ln n里的n是当前节点总访问次数,不是根节点的总访问次数
		score=exploit+scalar*explore
		if score==bestscore:
			bestchildren.append(c)
		if score>bestscore:
			bestchildren=[c]
			bestscore=score
	if len(bestchildren)==0:
		logger.warn("OOPS: no best child found, probably fatal")
	return random.choice(bestchildren)

# 功能: Simulation,从当前状态开始模拟，直到终止状态，并返回终止状态奖励(根据终止状态的value到达goal的近似程度计算得到)。 输入:state 当前状态.
def DEFAULTPOLICY(state):
	while state.terminal()==False:
		state=state.next_state()
	return state.reward()

# Backpropagation, 从终止状态向上回溯到根节点，并更新回溯路上每个节点的visits与reward
def BACKUP(node,reward):
	while node!=None:
		node.visits+=1
		node.reward+=reward
		node=node.parent
	return

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='MCTS research code')
	parser.add_argument('--num_sims', action="store", required=True, type=int)
	parser.add_argument('--levels', action="store", required=True, type=int, choices=range(State.NUM_TURNS+1))
	args=parser.parse_args()
	
	# 通过打印日志可以看出,mian中实现的是一个下棋过程,每一步都选择根节点的最优子节点print(而不是选择最优叶子节点)。总共走args.levels步。
	# 且每一步中都会以当前状态重新进行UCTSEARCH,且search的最大步数是"args.num_sims/(l+1)"
	current_node=Node(State())
	for l in range(args.levels):
		current_node=UCTSEARCH(args.num_sims/(l+1),current_node)
		print("level %d"%l)
		print("Num Children: %d"%len(current_node.children))
		for i,c in enumerate(current_node.children):
			print(i,c)
		print("Best Child: %s"%current_node.state)
		
		print("--------------------------------")	
			
	
