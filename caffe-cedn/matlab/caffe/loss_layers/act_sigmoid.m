function [act] = act_sigmoid(pred)

act = 1 ./ (1 + exp(-pred)); 
