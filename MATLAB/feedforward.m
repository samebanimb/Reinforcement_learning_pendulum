function a = feedforward (s, network)
% Ensure input s is a column vector
if size(s, 1) == 1
    s = s';  % Transpose to ensure correct multiplication
end
layer1 = network.layer1_w' * s + network.layer1_b';
alayers = max(layer1,0);

layer2 = network.layer2_w' * alayers + network.layer2_b';
alayers = max(layer2,0);

layer3 = network.layer3_w' * alayers + network.layer3_b;
alayer = tanh(layer3);
a = alayer * 5; 
end