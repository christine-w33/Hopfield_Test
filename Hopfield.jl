using PyPlot
using PyCall
@pyimport matplotlib.patches as patch

#Functions to make plots

#make plot after each iteration based on trained weights
function plot_image(input_states)

  black = "#000000"; white="#FFFFFF"
  squareSide = 0.03

  fig = figure(figsize=(5,4))
  ax = axes([0,0,6.5,6.5])

  # Plotting squares

  for i = 1:20
      coordinates = [i%5*squareSide, (4-i/5)*squareSide]
      if input_states[i]< 0
        color = black
      else
        color = white
      end
      square = patch.Rectangle(coordinates, squareSide, squareSide,
                                                    color = color, linewidth=0.5)
      ax[:add_patch](square)
  end

end

#calculate the energy of the system
function energy(update_states, weights)
  energy = 0
  for i = 1:m-1
    for j = i+1:m
      energy += -1 * update_states[i] * update_states[j] * weights[i,j]
    end
  end
  energy
end

#Train and store weights into 4*5 matrix titled weights
function train_weights(patterns)

  scalefactor = 1./n

  weights = zeros(m,m)
    for i = 1:m-1
        for j = i+1:m
          weights[i,j] = scalefactor * dot(patterns[:,i], patterns[:,j])
          weights[j,i] = weights[i,j]
        end
    end

  weights

end

#input pattern to be recognized (what percentage of noise can we introduce?)

#update activations. Must be in random order. This function must loop until convergence is detected.
function recall(states, weights)

  #set initial states
  states = convert(Array{Float64}, states)
  activations = zeros(m)
  convergence = false
  update_states = deepcopy(states)

  while convergence == false
    srand() #set random seed
    update_sequence = randperm(m)
    for i in update_sequence
      activations[i] = dot(vec(weights[i,:]), vec(update_states))
      if activations[i] >= 0 #convert new states to binary and check if they are the same as states for previous loops
        update_states[i] = 1.0
      else
        update_states[i] = -1.0
      end
      #println("energy:", energy(update_states, weights))
    end
    if states == update_states #check for convergence
      convergence = true
      #plot_image(states)
      break
    else #if no convergence, update the states according to activation weights and run the whole block again
      #println("still converging")
      states = deepcopy(update_states)
    end
  end

  states
end

function hamming_distance(test_states, final_states)
  error = 0.0
  for i in 1:m
    if test_states[i] != final_states[i]
      error += 1
    end
  end
  error
end

function addnoise(state, noise)
  srand() #set random seed
  switch = rand(1:20, noise)
  for i in switch
    if state[i] > 0
      state[i] = -1.0
    else
      state[i] = 1.0
    end
  end
  state
end

#states = [1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1]
#plot_image(states)
#plt[:show]()


function test_capacity()
#train weights for each randomly selected pattern added to the network.
  hamming_averages = [] #initialize empty list for output
  for capacity in 1:n
    srand() #set random seed
    chose = rand(randperm(n),capacity) #choose random set of patterns at each capacity
    patterns = letters[chose,:]
    weights = train_weights(patterns)
    #at each capacity test the network for recall with each pattern involved.
    error = 0.0
    for tests in chose #test all the patterns the weights were trained on
      test_states = letters[tests,:]
      final_states = recall(test_states, weights)
      error += hamming_distance(test_states, final_states)
    end
    hamming_average = error / capacity #the error at this capacity is the total hamming distance divided by the amount of patterns.
    println("Number of patterns: $capacity ; Hamming average: $hamming_average") #print the results for each capacity
    hamming_averages = push!(hamming_averages, hamming_average)
  end
  hamming_averages
end


function test_noise()
  #randomly choose 3 patterns and train network weights
  srand()
  chose2 = rand(1:n, 2) #initially set as 3, changed to 2 because tests indiciated this was network training pattern capacity
  patterns = letters[chose2,:]
  weights = train_weights(patterns)
  #iteratively add more noise to the test patterns
  hamming_averages = [] #initialize a list for output
  for i in 0:20
    error = 0.0
    #test all patterns the network is trained for at each noise level
    for letter in chose2
      test_states = addnoise(letters[letter,:], i)
      final_states = recall(test_states, weights)
      error += hamming_distance(test_states, final_states)
    end
    hamming_average = error / 2
    percent_noise = float(i)/20
    println("percentage noise: $percent_noise ; Hamming average: $hamming_average")
    hamming_averages = push!(hamming_averages, hamming_average)
  end
  hamming_averages
end

#nest test for noise within test for capacity to see how both factors affect the performance of the network
function test_both(capacity)
  #choose random patterns from all at each capacity
  srand() #set random seed
  chose = rand(1:n, capacity)
  patterns = letters[chose,:]
  weights = train_weights(patterns)
  hamming_averages = []
  for i in 0:20
    error = 0.0
    #test all the patterns at a given capacity for a given amount of noise
    for tests in chose
      test_states = addnoise(letters[tests,:], i)
      final_states = recall(test_states, weights)
      error += hamming_distance(test_states, final_states)
    end
    hamming_average = error / capacity
    percent_noise = float(i)/20
    println("Number of patterns: $capacity ; Percentage noise: $percent_noise ; Hamming average: $hamming_average")
    hamming_averages = push!(hamming_averages, hamming_average)
  end
  hamming_averages
end


#20 units. 5*5. Black is 1, white is -1
letters = [-1 1 1 1 1 -1 1 1 1 1 -1 1 1 1 1 -1 -1 1 1 1; #letter L
            -1 1 1 1 -1 -1 -1 1 -1 -1 -1 1 -1 1 -1 -1 1 1 1 -1; #letter M
            1 -1 -1 -1 1 1 -1 1 1 1 1 -1 1 1 1 1 -1 -1 -1 1; #letter C
            1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1; #letter O
            -1 -1 -1 -1 -1 1 1 -1 1 1 -1 1 -1 1 1 -1 -1 -1 1 1; #letter J
            1 -1 -1 -1 1 1 1 -1 1 1 1 1 -1 1 1 1 1 -1 1 1; #letter T
            1 -1 -1 -1 1 1 -1 1 1 1 1 -1 -1 1 1 1 -1 1 1 1; #letter F
            1 -1 1 -1 1 1 -1 1 -1 1 1 -1 -1 -1 1 1 -1 1 -1 1; #letter H
            1 -1 -1 -1 1 1 -1 1 -1 1 1 -1 -1 -1 1 1 -1 1 1 1; #letter P
            -1 1 1 1 -1 -1 1 -1 1 -1 -1 -1 1 -1 -1 -1 1 1 1 -1] #letter W
dimensions = size(letters)
n = dimensions[1]
m = dimensions[2]

#uncomment a function to run a test and see the results displayed in terminal
#test_capacity()
#test_noise()
#test_both()

#functions to plot results

function plot_capacity()
  outcome_array = test_capacity()
  for trial in 1:9
    outcome = test_capacity()
    outcome_array = hcat(outcome_array, outcome)
  end
  outcome_array = convert(Array{Float64,2}, outcome_array) #neccessary to calculate standard deviation
  y = [mean(outcome_array[x,:]) for x in 1:10]
  errs = [std(outcome_array[x,:]) for x in 1:10]
  p = plot(x,y,linestyle="-",label="Base Plot")
  errorbar(x, y, yerr=errs, fmt="o")
  xlabel("Number of Training Patterns")
  ylabel("Recall Error (Hamming Distance Averages)")
  title("Capacity: Recall Error vs Training Patterns")
end

function plot_noise()
  outcome_array = test_noise()
  for trial in 1:19
    outcome = test_noise()
    outcome_array = hcat(outcome_array, outcome) 
  end #although repitive, this loop could not be placed into a function due to problems parsing function into an input variable
  outcome_array = convert(Array{Float64,2}, outcome_array) #neccessary to calculate standard deviation
  x = [x/20 for x in 0:20]
  y = [mean(outcome_array[x,:]) for x in 1:21]
  errs = [std(outcome_array[x,:]) for x in 1:21]
  p = plot(x,y,linestyle="-",label="Base Plot")
  errorbar(x, y, yerr=errs, fmt="o")
  xlabel("Fraction of Input Noise")
  ylabel("Recall Error (Hamming Distance Averages)")
  title("Recall Error vs Input Noise Fraction")
end

function plot_both()
  for capacity in 1:10
    outcome_array = test_both()
    for trial in 1:19
      outcome = test_both()
      outcome_array = hcat(outcome_array, outcome)
    end
    outcome_array = convert(Array{Float64,2}, outcome_array) #neccessary to calculate standard deviation
    x = [x/20 for x in 0:20]
    y = [mean(outcome_array[x,:]) for x in 1:21]
    errs = [std(outcome_array[x,:]) for x in 1:21]
    p = plot(x,y,linestyle="-",label="Base Plot")
    errorbar(x, y, yerr=errs, fmt="o")
  xlabel("Fraction of Input Noise")
  ylabel("Recall Error (Hamming Distance Averages)")
  title("Recall Error vs Input Noise Fraction at Increasing Capacity")
  end
end

#function plot_energyGradient()


plot_capacity()
#plot_noise()
#plot_both()
#plot_energyGradient()

plt[:show]()
