Ts = 0.05;
Tf = 25;

% Observation info
obsInfo = rlNumericSpec([4 1],...
    LowerLimit=[-0.5 -inf -inf -inf  ]',...
    UpperLimit=[ 0.5 inf  inf inf]');

% Name and description are optional and not used by the software
obsInfo.Name = "observations";
obsInfo.Description = "distance, angle,velocity and angular velocity";

% Action info
actInfo = rlNumericSpec([1 1],...
    LowerLimit=-5 ,...
    UpperLimit=5);

actInfo.Name = "voltage";

env = rlSimulinkEnv("Env_Agent_Simulation",...
    "Env_Agent_Simulation/agent",...
    obsInfo,actInfo);

env.ResetFcn = @(in)myResetFunction(in);

validateEnvironment(env)

%%
rng(0)
%%
% Observation path
obsPath = [
    featureInputLayer(obsInfo.Dimension(1),Name="obsInLyr")
    fullyConnectedLayer(256)
    reluLayer
    fullyConnectedLayer(256)
    reluLayer(Name="obsPathOutLyr")
    ];

% Action path
actPath = [
    featureInputLayer(actInfo.Dimension(1),Name="actInLyr")
    fullyConnectedLayer(256)
    reluLayer(Name="actPathOutLyr")
    ];

% Common path
commonPath = [
    concatenationLayer(1,2,'Name','concat')
    fullyConnectedLayer(512)
    reluLayer
    fullyConnectedLayer(256)
    reluLayer
    fullyConnectedLayer(1,Name="QValue")
    ];

% Create the network object and add the layers
criticNet = dlnetwork();
criticNet = addLayers(criticNet,obsPath);
criticNet = addLayers(criticNet,actPath);
criticNet = addLayers(criticNet,commonPath);

% Connect the layers
criticNet = connectLayers(criticNet, ...
    "obsPathOutLyr","concat/in1");
criticNet = connectLayers(criticNet, ...
    "actPathOutLyr","concat/in2");

%plot(criticNet)

criticNet = initialize(criticNet);
summary(criticNet)

critic = rlQValueFunction(criticNet, ...
    obsInfo,actInfo, ...
    ObservationInputNames="obsInLyr", ...
    ActionInputNames="actInLyr");

actorNet = [
    featureInputLayer(obsInfo.Dimension(1))
    fullyConnectedLayer(400)
    reluLayer
    fullyConnectedLayer(300)
    reluLayer
    fullyConnectedLayer(actInfo.Dimension(1))
    tanhLayer % Constrain actions to [-1, 1]
    scalingLayer('Scale', actInfo.UpperLimit)
    ];
actorNet = dlnetwork(actorNet);
actorNet = initialize(actorNet);
summary(actorNet)
%figure
%plot(actorNet)
actor = rlContinuousDeterministicActor(actorNet,obsInfo,actInfo);
%%
agent = rlDDPGAgent(actor,critic);
agent.SampleTime = Ts;

agent.AgentOptions.TargetSmoothFactor = 0.005;
agent.AgentOptions.DiscountFactor = 0.99;
agent.AgentOptions.MiniBatchSize = 128;
agent.AgentOptions.ExperienceBufferLength = 1e6; 

%agent.AgentOptions.NoiseOptions.Variance = 0.1;
agent.AgentOptions.NoiseOptions.VarianceDecayRate = 1e-5;
agentOptions.NoiseOptions.StandardDeviation = 0.2;

agent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-03;
agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = Inf;
agent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-04;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = Inf;
%%
agent.AgentOptions.ResetExperienceBufferBeforeTraining = false;
%%
getAction(agent,{rand(obsInfo.Dimension)})
%%
trainOpts = rlTrainingOptions(...
    MaxEpisodes=150000, ...
    MaxStepsPerEpisode=ceil(Tf/Ts), ...
    ScoreAveragingWindowLength=20, ...
    Verbose=false, ...
    Plots="training-progress",...
    StopTrainingCriteria="GlobalStepCount",...
    StopTrainingValue=1200000, ...
    SaveAgentCriteria = "EpisodeReward", ...
    SaveAgentValue = 4000);
%%
doTraining = true;
 
if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else
    % Load the pretrained agent for the example.
    load("saved_agent.mat","agent")
end

%%
%simOpts = rlSimulationOptions(MaxSteps=ceil(Tf/Ts),StopOnError="on",NumSimulations=50);
%experiences = sim(env,saved_agent,simOpts);


%%
function in = myResetFunction(in)
% Reset function to place custom cart-pole environment into a random
% initial state.
x = (rand - 0.5) * 0.4 ;
%x= 0.0;
x_dot= 0.0;
alpha= (rand - 0.5)* 2 * pi;
%alpha= (rand - 0.5)* 2 * (15/180) * pi ;
alpha_dot= rand - 0.5;

% Return initial environment state variables as logged signals.
%InitialState = [x;alpha;x_dot;alpha_dot];
in = setBlockParameter(in, ...
    "Env_Agent_Simulation/x", ...
    'Value',num2str(x),...
    "Env_Agent_Simulation/alpha", ...
    'Value',num2str(alpha),...
    "Env_Agent_Simulation/x_dot", ...
    'Value',num2str(x_dot),...
    "Env_Agent_Simulation/alpha_dot", ...
    'Value',num2str(alpha_dot), ...
    "Env_Agent_Simulation/step_stabilization", ...
    'Value','1',...
    "Env_Agent_Simulation/first_volt", ...
    'Value','0');
end
