import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras.python import keras as K
from tensorflow.contrib.data.python.ops.dataset_ops import Dataset
from tensorflow.python.ops.rnn import *
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
import os


class TreeLSTMCell:
    def __init__(self,units=50,input_shape=(1,100)):
        self.units=units
        self.input_shape=input_shape
        with tf.variable_scope("TreeLSTM") :
            c_init=tf.get_variable(name="c",shape=input_shape,initializer=tf.random_normal_initializer(),trainable=False)
            h_init=tf.get_variable(name="h",shape=input_shape,initializer=tf.random_normal_initializer(),trainable=False)
        with tf.variable_scope("forget") as f:
            f_W = tf.get_variable(name="W_f", shape=(self.input_shape[-1],self.units),
                                  dtype=tf.float32,initializer=tf.random_normal_initializer())
            f_U = tf.get_variable(name="U_f", shape=(self.input_shape[-1],self.units),
                                  dtype=tf.float32,initializer=tf.random_normal_initializer())
            f_b = tf.get_variable(name="b_f", shape=(self.input_shape[0],self.units),
                                  dtype=tf.float32,initializer=tf.random_normal_initializer())
        with tf.variable_scope("input") as i:
            i_W = tf.get_variable(name="W_i", shape=(self.input_shape[-1], self.units),
                                  dtype=tf.float32,initializer=tf.random_normal_initializer())
            i_U = tf.get_variable(name="U_i", shape=(self.input_shape[-1], self.units),
                                  dtype=tf.float32,initializer=tf.random_normal_initializer())
            i_b = tf.get_variable(name="b_i", shape=(self.input_shape[0],self.units), dtype=tf.float32,
                                  initializer=tf.random_normal_initializer())
        with tf.variable_scope("output") as o:
            o_W = tf.get_variable(name="W_o", shape=(self.input_shape[-1], self.units), dtype=tf.float32,
                                  initializer=tf.random_normal_initializer())
            o_U = tf.get_variable(name="U_o", shape=(self.input_shape[-1], self.units), dtype=tf.float32,
                                  initializer=tf.random_normal_initializer())
            o_b = tf.get_variable(name="b_o", shape=(self.input_shape[0], self.units), dtype=tf.float32,
                                  initializer=tf.random_normal_initializer())

        with tf.variable_scope("new_state") as n:
            n_W = tf.get_variable(name="W_n", shape=(self.input_shape[1], self.units), dtype=tf.float32,
                                  initializer=tf.random_normal_initializer())
            n_U = tf.get_variable(name="U_n", shape=(self.input_shape[1], self.units), dtype=tf.float32,
                                  initializer=tf.random_normal_initializer())
            n_b = tf.get_variable(name="b_n", shape=(self.input_shape[0], self.units), dtype=tf.float32,
                                  initializer=tf.random_normal_initializer())

    def call(self,input,states):

        cond = lambda i, s, h_j_tile, f_j, forget_flow: tf.less(i, states.get_shape()[0])

        def body(i, s, h_j_tile, f_j, forget_flow):
            with tf.variable_scope("forget", reuse=True):
                h_k = tf.gather(s, i)[0]
                update_h_j_tile = tf.add(h_j_tile, h_k)

                W = tf.get_variable(name="W_f")
                U = tf.get_variable(name="U_f")
                b = tf.get_variable(name="b_f")

                f_j_k = tf.sigmoid(tf.matmul(input, W) + tf.matmul(h_k, U) + b)
                update_f_j = tf.add(f_j, f_j_k)

                update_forget_flow = tf.add(forget_flow, tf.multiply(f_j_k, tf.gather(s, i)[1]))

                update_i = tf.add(i, 1)

                return update_i, s, update_h_j_tile, update_f_j, update_forget_flow

        _, states, h_j_tile, f_j, update_forget_flow = tf.while_loop(cond, body,
                                                                     [tf.constant(0, dtype=tf.int32),
                                                                      states,
                                                                      tf.constant(np.zeros(self.input_shape), dtype=tf.float32),
                                                                      tf.constant(np.zeros((self.input_shape[0],self.units)),dtype=tf.float32),
                                                                      tf.constant(np.zeros((self.input_shape[0],self.units)),dtype=tf.float32)
                                                                      ])
        with tf.variable_scope("input", reuse=True):
            W = tf.get_variable(name="W_i")
            U = tf.get_variable(name="U_i")
            b = tf.get_variable(name="b_i")
            i_j = tf.sigmoid(tf.matmul(input, W) + tf.matmul(h_j_tile, U) + b)
        with tf.variable_scope("output", reuse=True):
            W = tf.get_variable(name="W_o")
            U = tf.get_variable(name="U_o")
            b = tf.get_variable(name="b_o")
            o_j = tf.sigmoid(tf.matmul(input, W) + tf.matmul(h_j_tile, U) + b)
        with tf.variable_scope("new_state", reuse=True):
            W = tf.get_variable(name="W_n")
            U = tf.get_variable(name="U_n")
            b = tf.get_variable(name="b_n")
            n_j = tf.sigmoid(tf.matmul(input, W) + tf.matmul(h_j_tile, U) + b)

        c_j = tf.multiply(i_j, n_j) + update_forget_flow
        h_j = tf.multiply(o_j, tf.tanh(c_j))
        return h_j, (h_j, c_j)

    def init_call(self,input):
        with tf.variable_scope("TreeLSTM", reuse=True):
            c = tf.get_variable("c")
            h = tf.get_variable("h")
            state = [(h, c)]
            stacked_state = tf.stack(state)
        h_j, s = self.call(input, stacked_state)
        return h_j,s


class Node:
    def __init__(self, input=None, label=None, shape=None,name=None, children=None, parent=None):
        if parent is not None:
            if type(parent) is Node:
                parent._children.append(self)
                self._parent = parent
        else:
            self._parent = parent

        if children is not None:
            if type(children) is Node:
                children._parent = self
                self._children.append(children)
        else:
            self._children = []

        if input is not None:
            self._input=input
            self._shape=self._input.get_shape()
        else:
            assert shape is not None
            if name is not None:
                self._input=tf.placeholder(name=name,dtype=tf.float32,shape=shape)
                self._shape=shape
            else:
                self._input = tf.placeholder( dtype=tf.float32, shape=shape)
                self._shape = shape


        self._predict = None
        if label is not None:
            self._label = label
        else:
            if name is not None:
                self._label = tf.placeholder(name=name+"_y", dtype=tf.float32, shape=shape)
            else:
                self._label = tf.placeholder(dtype=tf.float32, shape=shape)

        ## This one use to control the flow of the hidden state from children nodes to their parent
        self._children_hidden_list = []

        ##This one to extract pair (predict,label) after building the graph
        ##Use to compute loss
        self._pair = []

    @property
    def children(self):
        return self._children

    @property
    def parent(self):
        return self._parent

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._predict

    @output.setter
    def output(self,val):
        self._predict=val

    @property
    def label(self):
        return self._label

    @property
    def children_hidden_list(self):
        return self._children_hidden_list

    @property
    def pair(self):
        return self._pair

    @pair.setter
    def pair(self,list_):
        self._pair=list_

class TreeLSTM:
    def __init__(self,cell,root=None,curr=None):
        assert root is not None
        assert curr is not None

        self._root=root
        self._curr=curr
        self._cell=cell

    @property
    def root(self):
        return self._root

    @property
    def curr(self):
        return self._curr

    @property
    def isRoot(self):
        if self.curr == self.root:
            return True
        else:
            return False

    @property
    def cell(self):
        return self._cell

    @property
    def isExternal(self):
        if len(self._curr.children) > 0:
            return False
        else:
            return True

    @property
    def isInternal(self):
        if len(self._curr.children) > 0:
            return True
        else:
            return False

    def subtree(self, u):
        return TreeLSTM(cell=self.cell,root=self.root,curr=u)

    def traversal(self): #Using postorder traversal to update output each node
        for child in self.curr.children:
            self.subtree(child).traversal()

        if self.isRoot is True:
            stacked_states = tf.stack(self.curr.children_hidden_list)
            out, state = self.cell.call(self.curr.input, stacked_states)
            self.curr.output = out
            return
        elif self.isInternal is True:
            stacked_states = tf.stack(self.curr.children_hidden_list)
            out, state = self.cell.call(self.curr.input, stacked_states)
            self.curr.parent.children_hidden_list.append(state)
            self.curr.output = out
            return
        if self.isExternal is True:
            out, state = self.cell.init_call(self.curr.input)
            self.curr.parent.children_hidden_list.append(state)
            self.curr.output = out
            return

    def update_pairs(self):
        for child in self.curr.children:
            self.subtree(child).update_pairs()

        if self.isInternal is True:
            tmp_list = []
            for child in self.curr.children:
                tmp_list.extend(child.pair)
            tmp_list.append((self.curr.output, self.curr.label))
            self.curr.pair = tmp_list
            return
        elif self.isExternal is True:
            self.curr.pair.append((self.curr.output, self.curr.label))
            return

# EX1: each input will in shape (1,100)
#                   i1
#           i2_1        i2_2

# i1=tf.constant(np.random.rand(1,100),dtype=tf.float32)
# i2_1=tf.constant(np.random.rand(1,100),dtype=tf.float32)
# i2_2=tf.constant(np.random.rand(1,100),dtype=tf.float32)
#
# noise=tf.constant(np.random.rand(1,100),dtype=tf.float32)
#
# y1=i1*12-noise
# y2_1=i2_1*10-noise
# y2_2=i2_2*9+noise
#
#
# n1=Node(input=i1,label=y1)
# n2_1=Node(input=i2_1,label=y2_1,parent=n1)
# n2_2=Node(input=i2_2,label=y2_2,parent=n1)
#
# cell=TreeLSTMCell(units=100,input_shape=(1,100))
# tree=TreeLSTM(cell=cell,root=n1,curr=n1)
#
# tree.traversal()
# tree.update_pairs()
# pair=tree.curr.pair
# predicts=[p[0] for p in pair]
# label=[p[1]for p in pair]
# loss=tf.losses.sigmoid_cross_entropy(predicts,label)
#
# optimizer=tf.train.GradientDescentOptimizer(0.01)
# training_step=optimizer.minimize(loss)
#
# init=tf.initialize_all_variables()
# with tf.Session() as sess:
#     sess.run(init)
#     first_loss=sess.run(loss)
#     print("Loss: %f"%first_loss)
#     print("Optimizing...")
#     for i in range(100):
#         sess.run(training_step)
#         next_loss=sess.run(loss)
#         print("next loss : %f" %next_loss)


# EX2: A tree with different structure
#                    i3
#           i4_1            i4_2            i4_3
#     i5_1  i5_2  i5_3    i5_4          i5_5      i5_6
#                         i6_1  i6_2
#                       i7_1

##INPUT
i3=tf.constant(np.random.rand(1,100),dtype=tf.float32)

i4_1=tf.constant(np.random.rand(1,100),dtype=tf.float32)
i4_2=tf.constant(np.random.rand(1,100),dtype=tf.float32)
i4_3=tf.constant(np.random.rand(1,100),dtype=tf.float32)

i5_1=tf.constant(np.random.rand(1,100),dtype=tf.float32)
i5_2=tf.constant(np.random.rand(1,100),dtype=tf.float32)
i5_3=tf.constant(np.random.rand(1,100),dtype=tf.float32)
i5_4=tf.constant(np.random.rand(1,100),dtype=tf.float32)
i5_5=tf.constant(np.random.rand(1,100),dtype=tf.float32)
i5_6=tf.constant(np.random.rand(1,100),dtype=tf.float32)

i6_1=tf.constant(np.random.rand(1,100),dtype=tf.float32)
i6_2=tf.constant(np.random.rand(1,100),dtype=tf.float32)

i7_1=tf.constant(np.random.rand(1,100),dtype=tf.float32)



##OUTPUT
o3=tf.constant(np.random.rand(1,100),dtype=tf.float32)

o4_1=tf.constant(np.random.rand(1,100),dtype=tf.float32)
o4_2=tf.constant(np.random.rand(1,100),dtype=tf.float32)
o4_3=tf.constant(np.random.rand(1,100),dtype=tf.float32)

o5_1=tf.constant(np.random.rand(1,100),dtype=tf.float32)
o5_2=tf.constant(np.random.rand(1,100),dtype=tf.float32)
o5_3=tf.constant(np.random.rand(1,100),dtype=tf.float32)
o5_4=tf.constant(np.random.rand(1,100),dtype=tf.float32)
o5_5=tf.constant(np.random.rand(1,100),dtype=tf.float32)
o5_6=tf.constant(np.random.rand(1,100),dtype=tf.float32)

o6_1=tf.constant(np.random.rand(1,100),dtype=tf.float32)
o6_2=tf.constant(np.random.rand(1,100),dtype=tf.float32)

o7_1=tf.constant(np.random.rand(1,100),dtype=tf.float32)


##Create nodes
n3=Node(input=i3,label=o3)

n4_1=Node(input=i4_1,label=o4_1,parent=n3)
n4_2=Node(input=i4_2,label=o4_2,parent=n3)
n4_3=Node(input=i4_3,label=o4_3,parent=n3)

n5_1=Node(input=i5_1,label=o5_1,parent=n4_1)
n5_2=Node(input=i5_2,label=o5_2,parent=n4_1)
n5_3=Node(input=i5_3,label=o5_3,parent=n4_1)

n5_4=Node(input=i5_4,label=o5_4,parent=n4_2)

n5_5=Node(input=i5_5,label=o5_5,parent=n4_3)
n5_6=Node(input=i5_6,label=o5_6,parent=n4_3)

n6_1=Node(input=i6_1,label=o6_1,parent=n5_4)
n6_2=Node(input=i6_2,label=o6_2,parent=n5_4)

n7_1=Node(input=i7_1,label=o7_1,parent=n6_1)


##Create Tree
cell2=TreeLSTMCell(units=100)
tree2=TreeLSTM(cell=cell2,root=n3,curr=n3)

tree2.traversal()
tree2.update_pairs()

pair=tree2.curr.pair
predicts=[p[0] for p in pair ]
labels=[p[1] for p in pair]

loss2=tf.losses.sigmoid_cross_entropy(predicts,labels)
optimizer=tf.train.GradientDescentOptimizer(0.01)
training_step=optimizer.minimize(loss2)

init=tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    first_loss=sess.run(loss2)
    print("Loss: %f"%first_loss)
    print("Optimizing...")
    for i in range(100):
        sess.run(training_step)
        next_loss=sess.run(loss2)
        print("next loss : %f" %next_loss)
