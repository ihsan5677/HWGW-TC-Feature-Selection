import random
import numpy
import math

def logistic_map(X0, MU):
    return MU * X0 * (1 - X0)

def HWGWTC(objf, lb, ub, dim, SearchAgents_no, Max_iter, MU = 3.999):
    # Initialize the logistic map sequence
    logistic_seq = numpy.random.rand()
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
        
    # initialize position vector and score for the leader
    Leader_pos = numpy.zeros(dim)
    Leader_score = float("inf")  # change this to -inf for maximization problems
    Leader_pos2 = numpy.zeros(dim)
    Leader_score2 = float("inf")
    Leader_pos3 = numpy.zeros(dim)
    Leader_score3 = float("inf")
    
    # Initialize the positions of search agents
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        logistic_seq = logistic_map(logistic_seq, MU)
        Positions[:, i] = logistic_seq * (ub[i] - lb[i]) + lb[i]

    t = 0  # Loop counter
    # Main loop
    while t < Max_iter:
        for i in range(0, SearchAgents_no):
            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])
            
            # Calculate objective function for each search agent
            fitness = objf(Positions[i, :])
            
            # Update the leader
            if fitness < Leader_score:  # Change this to > for maximization problem
                Leader_score = fitness
                # Update alpha
                Leader_pos = Positions[i, :].copy()  # copy current whale position into the leader position
                
            if fitness > Leader_score and fitness < Leader_score2:  # Change this to > for maximization problem
                Leader_score2 = fitness
                # Update beta
                Leader_pos2 = Positions[i, :].copy()
                
            if fitness > Leader_score and fitness > Leader_score2 and fitness < Leader_score3:  # Change this to > for maximization problem
                Leader_score3 = fitness
                # Update delta
                Leader_pos3 = Positions[i, :].copy()
        
        # the cos function will add non-linearity to the parameter a.
        a =  2 * numpy.cos((numpy.pi * t) / (2 * Max_iter))
        
        a2 = -1 + t * ((-1) / Max_iter)
        rr1 = 2 - t * ((2) / Max_iter)
        
        # Update the logistic sequence
        logistic_seq = logistic_map(logistic_seq, MU)
        
        # Update the Position of search agents
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1 = random.random()  # r1 is a random number in [0,1]
                
                A = 2 * a * r1 - a 
                
                b = 1
                l = (a2 - 1) * random.random() + 1  
                p = random.random()  
                
                if p < 0.5:
                    rr2 = (2 * numpy.pi) * random.random()
                    rr3 = 2 * random.random()
                    
                    #Exploration
                    if abs(A) >= 1:
                        #Sine function of SCA
                        Positions[i, j] = Positions[i, j]+(rr1 * numpy.sin(rr2) *abs(rr3 * Leader_pos[j] - Positions[i, j]))
                    
                    #Exploitation
                    elif abs(A) < 1:
                        D_Leader = abs(Leader_pos[j] - Positions[i, j])
                        D_Leader2 = abs(Leader_pos2[j] - Positions[i, j])
                        D_Leader3 = abs(Leader_pos3[j] - Positions[i, j])
                        Positions[i, j] = Leader_pos[j] - A * ((D_Leader+D_Leader2+D_Leader3)/3)
                
                elif p >= 0.5:
                    
                    #Exploration
                    if abs(A) >= 1:
                        distance2Leader = abs(Leader_pos[j] - Positions[i, j])
                        Positions[i, j] = distance2Leader * math.exp(b * l) * math.cos(l * 2 * math.pi)+ Leader_pos[j]
                    
                    #Exploitation
                    elif abs(A) < 1:
                        D_Leader1 = abs(Leader_pos[j] - Positions[i, j])
                        D_Leader2 = abs(Leader_pos2[j] - Positions[i, j])
                        D_Leader3 = abs(Leader_pos3[j] - Positions[i, j])
                        distance2Leader = abs(Leader_pos[j] - Positions[i, j])
                        Positions[i, j] = (D_Leader1+D_Leader2+D_Leader3)/3
       
        if t % 1 == 0:
            print(["At iteration " + str(t) + " the best fitness is " + str(Leader_score)])
        t = t + 1
    return Leader_pos