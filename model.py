import numpy as np
import tensorflow as tf

class ActorCritic:
  def __init__(
          self,
          state_dims,
          action_dims,
          learning_rate_actor=0.001,
          learning_rate_critic=0.01,
          reward_decay=0.9,
          output_graph=False
  ):
    self.state_dims = state_dims
    self.action_dims = action_dims
    self.lra = learning_rate_actor
    self.lrc = learning_rate_critic
    self.gamma = reward_decay

    self.memory = []
    
    self._build_net()

    self.sess = tf.Session()

    if output_graph:
      tf.summary.FileWriter('logs/', self.sess.graph)

    self.sess.run(tf.global_variables_initializer())

  def _build_net(self):
    with tf.variable_scope('actor'):
      self.s = tf.placeholder(tf.float32, [None, self.state_dims], name='state')
      self.a = tf.placeholder(tf.int32, [None], name='actions')
      self.td_error = tf.placeholder(tf.float32, [None], name='td_error')

      with tf.variable_scope('network'):
        h1 = tf.layers.dense(self.s, 20, activation=tf.nn.relu)
        all_act = tf.layers.dense(h1, self.action_dims, activation=None)

      self.all_act_prob = tf.nn.softmax(all_act)

      with tf.variable_scope('loss'):
        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.a)
        self.aloss = tf.reduce_mean(neg_log_prob * self.td_error)
      with tf.variable_scope('train'):
        self.atrain_op = tf.train.AdamOptimizer(self.lra).minimize(self.aloss)

    with tf.variable_scope('critic'):
      self.s_ = tf.placeholder(tf.float32, [None, self.state_dims], name='nstate')
      self.r = tf.placeholder(tf.float32, [None], name='reward')

      with tf.variable_scope('network'):
        h1 = tf.layers.dense(self.s_, 20, activation=tf.nn.relu)
        self.v = tf.layers.dense(h1, 1, activation=None)

      with tf.variable_scope('squared_TD_error'):
        ctd_error = tf.reshape( self.v, [-1] ) - self.r
        self.closs = tf.reduce_mean(tf.square(ctd_error))
      with tf.variable_scope('train'):
        self.ctrain_op = tf.train.AdamOptimizer(self.lrc).minimize(self.closs)

  def store_transition(self, s, a, r, ns):
    self.memory.append( (s, a, r, ns) )

  def choose_action(self, state, _eval=False):
    state = state[np.newaxis, :]

    action_prob = self.sess.run(self.all_act_prob, feed_dict={
                      self.s: state })
    if _eval:
      action = np.argmax( action_prob )
    else:
      action = np.random.choice(self.action_dims, p=action_prob[0])

    return action

  def learn(self):
    s  = [ m[0] for m in self.memory ]
    a  = [ m[1] for m in self.memory ]
    r = self._discount_and_norm_reward()
    ns = [ m[3] for m in self.memory ]
    
    # update critic
    self.sess.run(self.ctrain_op, feed_dict={
        self.s_: s,
        self.r: r })

    nv = self.sess.run(self.v, feed_dict={ self.s_: ns })
    v = self.sess.run(self.v, feed_dict={ self.s_: s })

    # update actor
    self.sess.run(self.atrain_op, feed_dict={
        self.s: s,
        self.a: a,
        self.td_error: np.reshape(nv-v, -1)})

  def _discount_and_norm_reward(self):
    discounted_ep_rs = np.zeros(len(self.memory))
    cummluative_reward = 0
    for t in reversed(range(0, len(self.memory))):
      cummluative_reward = self.memory[t][2] + cummluative_reward * self.gamma
      discounted_ep_rs[t] = cummluative_reward

    discounted_ep_rs -= np.mean(discounted_ep_rs)
    discounted_ep_rs /= np.std(discounted_ep_rs)

    return discounted_ep_rs
