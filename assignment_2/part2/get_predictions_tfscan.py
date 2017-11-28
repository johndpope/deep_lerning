    def _model_step_given_phrase(self, state_tuple, x):
        state           = state_tuple[1]
        output, state   = self.stacked_lstm(x, state)
        return (output,state)

    def _model_step_predict(self, state_tuple, x):
        logits_prev   = state_tuple[0]
        state         = state_tuple[1]
        output, state = self.stacked_lstm(logits_prev, state)
        logits        = tf.matmul(output, self.Wout) + self.bias_out
        probs         = self.probabilities(logits)
        preds         = self.predictions(probs)
        preds         = tf.one_hot(preds, self._vocab_size, axis=-1)
        return (preds,state)

    def get_predictions(self, phrase, prediction_length):
        # Get state after phrase
        zero_state = self.stacked_lstm.zero_state(1, tf.float32)
        output     = tf.zeros([1, self._lstm_num_hidden], tf.float32)
        init_state = (output,zero_state)
        states     = tf.scan(self._model_step_given_phrase, phrase, initializer=init_state)

        # Get next states
        output     = states[0][-1]
        last_state = states[-1]
        logits     = tf.matmul(output, self.Wout) + self.bias_out
        probs      = self.probabilities(logits)
        preds      = self.predictions(probs)
        preds      = tf.one_hot(preds, self._vocab_size, axis=-1)
        #init_state = (preds,last_state)
        #new_states = tf.scan(self._model_step_predict, np.array(range(prediction_length)), initializer=init_state)