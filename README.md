# Deep Learning Specialization: Sequence Models (Course 5) - Programming Assignments

This repository contains implementations of various Recurrent Neural Network (RNN) architectures, LSTM cells, and their corresponding forward and backward propagation algorithms built from scratch using Python and NumPy. These assignments are part of the **Sequence Models** course in the Deep Learning Specialization (Coursera).

---

##  Project Structure & Exercises

The project covers 8 core exercises focusing on building sequence models step-by-step:

### **Part 1: Recurrent Neural Networks (RNNs)**
1. **`rnn_cell_forward` (Exercise 1):** Implements a single forward step of a standard RNN cell, calculating the next hidden state ($\tanh$) and the output prediction ($\text{softmax}$).
2. **`rnn_forward` (Exercise 2):** Implements forward propagation for an entire RNN sequence over $T_x$ time steps.
3. **`rnn_cell_backward` (Exercise 3):** Implements the backward pass (single time-step) for an RNN cell using the chain rule and $\tanh$ derivative.
4. **`rnn_backward` (Exercise 4):** Implements Backpropagation Through Time (BPTT) for a full RNN sequence.

### **Part 2: Long Short-Term Memory (LSTM)**
5. **`lstm_cell_forward` (Exercise 5):** Implements a single forward step of an LSTM cell, including forget gate, update gate, memory state, and output gate mechanics.
6. **`lstm_forward` (Exercise 6):** Implements forward propagation for an LSTM network across multiple time steps ($T_x$).
7. **`lstm_cell_backward` (Exercise 7):** Implements the backward pass for a single LSTM cell step, calculating gradients for gates, parameters, and previous states.
8. **`lstm_backward` (Exercise 8):** Implements the full backward pass (BPTT) for an LSTM network over an entire input sequence.

---
