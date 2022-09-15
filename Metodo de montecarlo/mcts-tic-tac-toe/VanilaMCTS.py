from time import sleep
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class policy(object):
    def __init__(self):
        self.tree = {}
        pass


class VanilaMCTS(object):
    def __init__(self, n_iterations=50, depth=15, exploration_constant=5.0, tree = None, win_mark=3, game_board=None, player=None):
        self.n_iterations = n_iterations
        self.depth = depth
        self.exploration_constant = exploration_constant
        self.total_n = 0

        self.leaf_node_id = None

        n_rows = len(game_board)
        self.n_rows = n_rows
        self.win_mark = win_mark

        if tree == None:
            self.tree = self._set_tictactoe(game_board, player)
        else:
            self.tree = tree

    #Inicializando el arbol
    def _set_tictactoe(self, game_board, player):
        root_id = (0,)
        tree = {root_id: {'state': game_board,
                          'player': player,
                          'child': [],
                          'parent': None,
                          'n': 0,
                          'w': 0,
                          'q': None}}
        return tree

    def selection(self):
        '''
       Se comienza en el nodo raiz y seleccionamos el nodo secundario sucesivo con el valor UCB mas alto hasta llegar a uno sin elementos
        in :
        - arbol
        return:
        - el id del nodo
        - profundidad del arbol (desde el nodo raiz)
        '''
        #Encontrar el nodo sin hijos
        leaf_node_found = False
        #Nodo raiz
        leaf_node_id = (0,) 
        
        while not leaf_node_found:
            node_id = leaf_node_id
            #Muestra la cantidad de hijos que tiene el nodo raiz
            n_child = len(self.tree[node_id]['child'])
            #Encontro que el nodo raiz no tiene hijos y sale
            if n_child == 0:
                leaf_node_id = node_id
                leaf_node_found = True
            #El nodo raiz tiene hijos y los va a recorrer todos
            else:

                maximum_uct_value = -100.0
                
                for i in range(n_child):
                    #hace referencia al nodo donde esta
                    action = self.tree[node_id]['child'][i]
                    # print('leaf_node_id', leaf_node_id)
                    child_id = node_id + (action,)
                    #calcular la formula uct que sirve para ver cual es el mejor nodo para eligir
                    #Numero de ganancias simuladas de ese nodo 
                    w = self.tree[child_id]['w']
                    # número de simulaciones que han ocurrido para ese nodo
                    n = self.tree[child_id]['n']
                    #Numero  de visitas para el nodo principal
                    total_n = self.total_n
                    if n == 0:
                        n = 1e-4
                    #Formula de UCT
                    #explotación 
                    exploitation_value = w / n
                    #exploracion
                    exploration_value  = np.sqrt(np.log(total_n)/n)
                    #exploration_constant mantiene el equillibrio entre ser nodos con buena alta probabilidad de ser elegidos y los que casi no se han visitado
                    uct_value = exploitation_value + self.exploration_constant * exploration_value
                    if uct_value > maximum_uct_value:
                        maximum_uct_value = uct_value
                        leaf_node_id = child_id

        depth = len(leaf_node_id) # profundidad del nodo

        return leaf_node_id, depth
    
    def expansion(self, leaf_node_id):
        '''
        Crea todo los resultados posibles del nodo hoja(nodo que no tiene hijos)
        in: tree, leaf_node
        out: Arbol expandido (self.tree),
             nodo hijo seleccionado aleatoriamente (child_node_id)
        '''
        #Nodo hoja
        leaf_state = self.tree[leaf_node_id]['state']
        #Determina si ya alguien gano el juego
        winner = self._is_terminal(leaf_state)
        #Los posibles estados que va jugar
        possible_actions = self._get_valid_actions(leaf_state)
        child_node_id = leaf_node_id 

        if winner is None:
            '''
            cuando el estado de la hoja no es el estado final
            '''
            childs = []
            #Recorre todos las posibles resultados que tiene para jugar
            for action_set in possible_actions:
                #action es la ubicacion que queda en el tablero
                #action_idx el arbol donde se encuetra esa accion
                action, action_idx = action_set

                #copia el estado actual del la matriz
                state = deepcopy(self.tree[leaf_node_id]['state'])
                #El jugador que esta jugando actualmente
                current_player = self.tree[leaf_node_id]['player']

                if current_player == 'o':
                    next_turn = 'x'
                    state[action] = 1
                    
                else:
                    next_turn = 'o'
                    state[action] = -1
                #nodo hijo
                child_id = leaf_node_id + (action_idx, )
                #lista de nodos
                childs.append(child_id)
                #se agrega un nodo al arbol
                self.tree[child_id] = {'state': state,
                                       'player': next_turn,
                                       'child': [],
                                       'parent': leaf_node_id,
                                       'n': 0, 'w': 0, 'q':0}
                self.tree[leaf_node_id]['child'].append(action_idx)
            #Se genera un numero random entre todos los nodos hijos
            rand_idx = np.random.randint(low=0, high=len(childs), size=1)
            #nodo random
            child_node_id = childs[rand_idx[0]]
        return child_node_id
    #esta funcion dice que jugador gano el juego
    def _is_terminal(self, leaf_state):
        '''
        
        in: game state
        out: Quien gano? ('o', 'x', 'Empate', None)
             (None = juego no terminado)
        '''
        def __who_wins(sums, win_mark):
            if np.any(sums == win_mark):
                return 'o'
            if np.any(sums == -win_mark):
                return 'x'
            return None
        #Funcion que recorre las filas,columnas y diagonal para determinar quien ganó
        def __is_terminal_in_conv(leaf_state, win_mark):
            # revisa row/col
            for axis in range(2):

                sums = np.sum(leaf_state, axis=axis)
                result = __who_wins(sums, win_mark)
                if result is not None:
                    return result
            # revisa diagonal
            for order in [-1,1]:
                diags_sum = np.sum(np.diag(leaf_state[::order]))
                result = __who_wins(diags_sum, win_mark)
                if result is not None:
                    return result
            return None
        #tiene que tener 3 marcas
        win_mark = self.win_mark
        #numero de filas del tablero
        n_rows_board = len(self.tree[(0,)]['state'])
        window_size = win_mark
        window_positions = range(n_rows_board - win_mark + 1)
        #recorre filas y columnas
        for row in window_positions:
            for col in window_positions:
                window = leaf_state[row:row+window_size, col:col+window_size]                
                winner = __is_terminal_in_conv(window, win_mark)
                if winner is not None:
                    return winner
        
        if not np.any(leaf_state == 0):
            '''
            es empate
            '''
            return 'empate'
        return None
    def _get_valid_actions(self, leaf_state):
        '''
        devuelve todas las posibles acciones
        in:
        - leaf_state
        out:
        - posibles acciones ((row,col), action_idx)
        '''
        actions = []
        count = 0
        state_size = len(leaf_state)
        #Coloca en una lista todas las posiciones donde esten vacias
        for i in range(state_size):
            for j in range(state_size):
                if leaf_state[i][j] == 0:
                    actions.append([(i, j), count])
                count += 1

        return actions

    def simulation(self, child_node_id):
        '''
        simula el juego desde el nodo hijo hasta que termina
        in:
        - id del nodo hijo seleccionado al azar
        out:
        - ganador ('o', 'x', 'emapate')
        '''
        self.total_n += 1
        #copia el tablero del juego actual
        state = deepcopy(self.tree[child_node_id]['state'])
        #copia el ultimo jugador
        previous_player = deepcopy(self.tree[child_node_id]['player'])
        anybody_win = False
        #simula hasta que haya un ganador
        while not anybody_win:
            winner = self._is_terminal(state)
            if winner is not None:
                anybody_win = True
            else:
                possible_actions = self._get_valid_actions(state)
                # aleatorio 
                rand_idx = np.random.randint(low=0, high=len(possible_actions), size=1)[0]
                #Elige una posicion de donde posiblemente puede mover
                action, _ = possible_actions[rand_idx]

                if previous_player == 'o':
                    current_player = 'x'
                    state[action] = -1
                else:
                    current_player = 'o'
                    state[action] = 1

                previous_player = current_player
        return winner

    def backprop(self, child_node_id, winner):
        #Hace una copia profunda del arbol
        player = deepcopy(self.tree[(0,)]['player'])
        #Valor acumulado del nodo hijo
        if winner == 'empate':
            reward = 0
        elif winner == player:
            reward = 1
        else:
            reward = -1

        finish_backprob = False
        node_id = child_node_id
        while not finish_backprob:
            #Contador de veces que ejecuto ese nodo
            self.tree[node_id]['n'] += 1
            #Se suma el valor acumulado
            self.tree[node_id]['w'] += reward
            #Puntuacion de confianza
            self.tree[node_id]['q'] = self.tree[node_id]['w'] / self.tree[node_id]['n']
            parent_id = self.tree[node_id]['parent']
            if parent_id == (0,):
                 #Contador de veces que ejecuto ese nodo
                self.tree[parent_id]['n'] += 1
                #Se suma el valor acumulado
                self.tree[parent_id]['w'] += reward
                #Puntuacion de confianza
                self.tree[parent_id]['q'] = self.tree[parent_id]['w'] / self.tree[parent_id]['n']
                finish_backprob = True
            else:
                node_id = parent_id

    def solve(self):
        for i in range(self.n_iterations):
            leaf_node_id, depth_searched = self.selection()
            child_node_id = self.expansion(leaf_node_id)
            winner = self.simulation(child_node_id)
            self.backprop(child_node_id, winner)
            if depth_searched > self.depth:
                break

        # SELECT BEST ACTION
        current_state_node_id = (0,)
        action_candidates = self.tree[current_state_node_id]['child']

        # qs = [self.tree[(0,)+(a,)]['q'] for a in action_candidates]
        best_q = -100
        for a in action_candidates:
            q = self.tree[(0,)+(a,)]['q']
            if q > best_q:
                best_q = q
                best_action = a

        # FOR DEBUGGING
        print('\n----------------------')
        print(' [-] game board: ')
        for row in self.tree[(0,)]['state']:
            print (row)
        print(' [-] person to play: ', self.tree[(0,)]['player'])
        print('\n [-] best_action: %d' % best_action)
        print(' best_q = %.2f' % (best_q))
        print(' [-] searching depth = %d' % (depth_searched))

        # FOR DEBUGGING        print(leaf_node_id)
        
        fig = plt.figure(figsize=(5,5))
        for a in action_candidates:
            # print('a= ', a)
            _node = self.tree[(0,)+(a,)]
            _state = deepcopy(_node['state'])

            _q = _node['q']
            _action_onehot = np.zeros(len(_state)**2)
            # _state[_action_onehot] = -1

            # print('action = %d, q = %.3f' % (a, _q))
            # print('state after action: ')
            # for _row in _state:
            #     print(_row)
            plt.subplot(len(_state),len(_state),a+1)
            plt.pcolormesh(_state, alpha=0.7, cmap="RdBu")
            plt.axis('equal')
            plt.gca().invert_yaxis()
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title('[%d] q=%.2f' % (a,_q))
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close(fig)


        return best_action, best_q, depth_searched



